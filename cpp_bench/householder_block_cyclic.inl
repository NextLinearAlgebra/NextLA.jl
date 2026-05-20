// Block-cyclic Path (h): Pz=1, b×b tiles, host gather/scatter + replicated Householder panel
// (Dgeqrf/Dorgqr) + host trailing (double or mixed-precision like scqr3_block_cyclic.inl).

#ifndef HOUSEHOLDER_BLOCK_CYCLIC_INL
#define HOUSEHOLDER_BLOCK_CYCLIC_INL

#include "block_cyclic.hpp"
#include "comm_groups.cuh"
#include "bench_vendor_metrics.hpp"
#include "nextla_mp_trail.hpp"

static int run_householder_bc_main(const Args& A, int P, int Px, int Py, int my_px, int my_py, int my_pz) {
    (void)P;
    int N = A.N, b = A.b;
    if (A.pz != 1) {
        if (_rank == 0) fprintf(stderr, "householder blockcyclic: require Pz=1\n");
        return 2;
    }
    if (my_pz != 0) {
        if (_rank == 0) fprintf(stderr, "householder blockcyclic: unexpected my_pz!=0\n");
        return 2;
    }
    if (!block_cyclic_valid(N, b, b, Px, Py)) {
        if (_rank == 0) fprintf(stderr, "householder blockcyclic: need N%%b==0, positive grid\n");
        return 3;
    }
    if (total_elements_bc(N, b, b, Px, Py) != (std::int64_t)N * N) {
        if (_rank == 0) fprintf(stderr, "householder blockcyclic: element count mismatch\n");
        return 4;
    }
    if (A.strict_b && !b_in_window(b, 1, N, Px, Py)) {
        if (_rank == 0)
            fprintf(stderr, "householder blockcyclic: b=%d violates §A3b window for c=1, N=%d pr=%d pc=%d\n", b, N, Px,
                    Py);
        return 55;
    }

    const bool full_f = (A.matrix == MatrixMode::FP32_FULL);
    const bool mp_trail = nextla_is_mp_trail_matrix(A.matrix);
    const bool use_tf32_trail = nextla_requests_tf32_matrix(A.matrix) && (NEXTLA_BENCH_HAVE_CUBLAS_TF32 != 0);
    if (nextla_requests_tf32_matrix(A.matrix) && !use_tf32_trail) {
        if (_rank == 0)
            fprintf(stderr, "householder blockcyclic: fp64mp_tf32 requires CUDA 11+ cuBLAS TF32.\n");
        return 91;
    }
    if (mp_trail && nextla_requests_tf32_matrix(A.matrix) && _rank == 0) {
        fprintf(stdout,
                "[note] householder blockcyclic: fp64mp_tf32 trailing uses host float W reduction (not cuBLAS TF32 "
                "tensor GEMMs on tiles); slab uses TF32 where available.\n");
        fflush(stdout);
    }
    if (full_f && (A.lookahead || A.n_ir > 0)) {
        if (_rank == 0)
            fprintf(stderr, "householder blockcyclic: fp32full with --la/--ir not wired; use slab.\n");
        return 93;
    }

    std::int64_t locr = numroc(N, b, my_px, 0, Px);
    std::int64_t locc = numroc(N, b, my_py, 0, Py);

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    nextla_maybe_print_tf32_trailing_banner_rank0(_rank, use_tf32_trail);

    P1ZsliceNccl p1{};
    create_p1_zslice_nccl_full(MPI_COMM_WORLD, _rank, Px, Py, 1, my_px, my_py, my_pz, &p1);

    cudaStream_t stream_hbc = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_hbc));
    cublasHandle_t cublas{};
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream_hbc));
    cusolverDnHandle_t cusolver{};
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream_hbc));

    double* d_loc_d = nullptr;
    float* d_loc_f = nullptr;
    size_t loc_sz = (size_t)std::max<std::int64_t>(1, locr) * (size_t)std::max<std::int64_t>(1, locc);
    if (full_f)
        CUDA_CHECK(cudaMalloc(&d_loc_f, loc_sz * sizeof(float)));
    else
        CUDA_CHECK(cudaMalloc(&d_loc_d, loc_sz * sizeof(double)));

    auto seed_local = [&]() {
        if (full_f) {
            std::vector<float> h(loc_sz);
            std::mt19937_64 rng(7 + _rank);
            std::normal_distribution<float> nrm(0.f, 1.f);
            for (std::int64_t lj = 0; lj < locc; ++lj) {
                for (std::int64_t li = 0; li < locr; ++li) {
                    int gi, gj;
                    local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                    h[(size_t)li + (size_t)lj * locr] = nrm(rng);
                }
            }
            CUDA_CHECK(cudaMemcpy(d_loc_f, h.data(), loc_sz * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            std::vector<double> h(loc_sz);
            std::mt19937_64 rng(7 + _rank);
            std::normal_distribution<double> nrm(0.0, 1.0);
            for (std::int64_t lj = 0; lj < locc; ++lj) {
                for (std::int64_t li = 0; li < locr; ++li) {
                    int gi, gj;
                    local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                    h[(size_t)li + (size_t)lj * locr] = nrm(rng);
                }
            }
            CUDA_CHECK(cudaMemcpy(d_loc_d, h.data(), loc_sz * sizeof(double), cudaMemcpyHostToDevice));
        }
    };
    seed_local();

    double* d_panel_d = nullptr;
    float* d_panel_f = nullptr;
    double* d_tau = nullptr;
    float* d_tau_f = nullptr;
    double* d_potw = nullptr;
    float* d_potwf = nullptr;
    int* d_info = nullptr;
    int lwork_geqrf = 0, lwork_orgqr = 0, lwork_geqrf_f = 0, lwork_orgqr_f = 0;
    if (full_f) {
        CUDA_CHECK(cudaMalloc(&d_panel_f, (size_t)N * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tau_f, (size_t)b * sizeof(float)));
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cusolver, N, b, d_panel_f, N, &lwork_geqrf_f));
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cusolver, N, b, b, d_panel_f, N, d_tau_f, &lwork_orgqr_f));
        int lw = std::max(lwork_geqrf_f, lwork_orgqr_f);
        CUDA_CHECK(cudaMalloc(&d_potwf, (size_t)lw * sizeof(float)));
    } else {
        CUDA_CHECK(cudaMalloc(&d_panel_d, (size_t)N * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_tau, (size_t)b * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolver, N, b, d_panel_d, N, &lwork_geqrf));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolver, N, b, b, d_panel_d, N, d_tau, &lwork_orgqr));
        int lw = std::max(lwork_geqrf, lwork_orgqr);
        CUDA_CHECK(cudaMalloc(&d_potw, (size_t)lw * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    int lwork_panel = full_f ? std::max(lwork_geqrf_f, lwork_orgqr_f) : std::max(lwork_geqrf, lwork_orgqr);

    std::vector<double> h_loc_d;
    std::vector<float> h_loc_f;
    std::vector<double> h_panel_d;
    std::vector<float> h_panel_f;
    if (full_f) {
        h_loc_f.resize(loc_sz);
        h_panel_f.resize((size_t)N * b);
    } else {
        h_loc_d.resize(loc_sz);
        h_panel_d.resize((size_t)N * b);
    }

    const double one_d = 1.0, zero_d = 0.0, neg_one_d = -1.0;
    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;

    auto run_qr_bc = [&]() {
        for (int k = 0; k < N; k += b) {
            int sb = std::min(b, N - k);
            if (full_f) {
                CUDA_CHECK(cudaMemcpy(h_loc_f.data(), d_loc_f, loc_sz * sizeof(float), cudaMemcpyDeviceToHost));
                std::fill(h_panel_f.begin(), h_panel_f.end(), 0.f);
                for (int t = 0; t < sb; ++t) {
                    int gj = k + t;
                    for (int gi = 0; gi < N; ++gi) {
                        std::int64_t li, lj;
                        if (global_to_local_bc(gi, gj, b, Px, Py, my_px, my_py, li, lj))
                            h_panel_f[(size_t)gi + (size_t)t * N] =
                                h_loc_f[(size_t)li + (size_t)lj * locr];
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, h_panel_f.data(), (int)(N * sb), MPI_FLOAT, MPI_SUM,
                              MPI_COMM_WORLD);
                CUDA_CHECK(
                    cudaMemcpy(d_panel_f, h_panel_f.data(), (size_t)N * sb * sizeof(float), cudaMemcpyHostToDevice));
                CUSOLVER_CHECK(cusolverDnSgeqrf(cusolver, N, sb, d_panel_f, N, d_tau_f, d_potwf, lwork_panel, d_info));
                CUSOLVER_CHECK(
                    cusolverDnSorgqr(cusolver, N, sb, sb, d_panel_f, N, d_tau_f, d_potwf, lwork_panel, d_info));
                CUDA_CHECK(cudaMemcpy(h_panel_f.data(), d_panel_f, (size_t)N * sb * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                for (int t = 0; t < sb; ++t) {
                    int gj = k + t;
                    for (int gi = 0; gi < N; ++gi) {
                        std::int64_t li, lj;
                        if (global_to_local_bc(gi, gj, b, Px, Py, my_px, my_py, li, lj))
                            h_loc_f[(size_t)li + (size_t)lj * locr] = h_panel_f[(size_t)gi + (size_t)t * N];
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_loc_f, h_loc_f.data(), loc_sz * sizeof(float), cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK(cudaMemcpy(h_loc_d.data(), d_loc_d, loc_sz * sizeof(double), cudaMemcpyDeviceToHost));
                std::fill(h_panel_d.begin(), h_panel_d.end(), 0.0);
                for (int t = 0; t < sb; ++t) {
                    int gj = k + t;
                    for (int gi = 0; gi < N; ++gi) {
                        std::int64_t li, lj;
                        if (global_to_local_bc(gi, gj, b, Px, Py, my_px, my_py, li, lj))
                            h_panel_d[(size_t)gi + (size_t)t * N] =
                                h_loc_d[(size_t)li + (size_t)lj * locr];
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, h_panel_d.data(), (int)(N * sb), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                CUDA_CHECK(cudaMemcpy(d_panel_d, h_panel_d.data(), (size_t)N * sb * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CUSOLVER_CHECK(cusolverDnDgeqrf(cusolver, N, sb, d_panel_d, N, d_tau, d_potw, lwork_panel, d_info));
                CUSOLVER_CHECK(
                    cusolverDnDorgqr(cusolver, N, sb, sb, d_panel_d, N, d_tau, d_potw, lwork_panel, d_info));
                CUDA_CHECK(cudaMemcpy(h_panel_d.data(), d_panel_d, (size_t)N * sb * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                for (int t = 0; t < sb; ++t) {
                    int gj = k + t;
                    for (int gi = 0; gi < N; ++gi) {
                        std::int64_t li, lj;
                        if (global_to_local_bc(gi, gj, b, Px, Py, my_px, my_py, li, lj))
                            h_loc_d[(size_t)li + (size_t)lj * locr] = h_panel_d[(size_t)gi + (size_t)t * N];
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_loc_d, h_loc_d.data(), loc_sz * sizeof(double), cudaMemcpyHostToDevice));
            }

            int ntrail = N - k - sb;
            if (ntrail <= 0) continue;

            if (full_f) {
                std::vector<float> h_W((size_t)sb * ntrail, 0.f);
                for (int t = 0; t < sb; ++t) {
                    for (int jtr = 0; jtr < ntrail; ++jtr) {
                        int gj = k + sb + jtr;
                        float acc = 0.f;
                        for (int gi = 0; gi < N; ++gi) {
                            float qv = h_panel_f[(size_t)gi + (size_t)t * N];
                            std::int64_t li, lj;
                            if (global_to_local_bc(gi, gj, b, Px, Py, my_px, my_py, li, lj))
                                acc += qv * h_loc_f[(size_t)li + (size_t)lj * locr];
                        }
                        h_W[(size_t)t + (size_t)jtr * sb] = acc;
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, h_W.data(), sb * ntrail, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                for (std::int64_t lj = 0; lj < locc; ++lj) {
                    for (std::int64_t li = 0; li < locr; ++li) {
                        int gi, gj;
                        local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                        if (gj < k + sb) continue;
                        int jtr = gj - k - sb;
                        float sum = 0.f;
                        for (int t = 0; t < sb; ++t)
                            sum += h_panel_f[(size_t)gi + (size_t)t * N] * h_W[(size_t)t + (size_t)jtr * sb];
                        h_loc_f[(size_t)li + (size_t)lj * locr] -= sum;
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_loc_f, h_loc_f.data(), loc_sz * sizeof(float), cudaMemcpyHostToDevice));
            } else if (mp_trail) {
                std::vector<float> h_Wf((size_t)sb * ntrail, 0.f);
                for (int t = 0; t < sb; ++t) {
                    for (int jtr = 0; jtr < ntrail; ++jtr) {
                        int gj = k + sb + jtr;
                        float acc = 0.f;
                        for (int gi = 0; gi < N; ++gi) {
                            double qv = h_panel_d[(size_t)gi + (size_t)t * N];
                            std::int64_t li, lj;
                            if (global_to_local_bc(gi, gj, b, Px, Py, my_px, my_py, li, lj))
                                acc += (float)(qv * h_loc_d[(size_t)li + (size_t)lj * locr]);
                        }
                        h_Wf[(size_t)t + (size_t)jtr * sb] = acc;
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, h_Wf.data(), sb * ntrail, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                for (std::int64_t lj = 0; lj < locc; ++lj) {
                    for (std::int64_t li = 0; li < locr; ++li) {
                        int gi, gj;
                        local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                        if (gj < k + sb) continue;
                        int jtr = gj - k - sb;
                        double sum = 0.0;
                        for (int t = 0; t < sb; ++t) {
                            double wv = (double)h_Wf[(size_t)t + (size_t)jtr * sb];
                            sum += wv * h_panel_d[(size_t)gi + (size_t)t * N];
                        }
                        h_loc_d[(size_t)li + (size_t)lj * locr] -= sum;
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_loc_d, h_loc_d.data(), loc_sz * sizeof(double), cudaMemcpyHostToDevice));
            } else {
                std::vector<double> h_W((size_t)sb * ntrail, 0.0);
                for (int t = 0; t < sb; ++t) {
                    for (int jtr = 0; jtr < ntrail; ++jtr) {
                        int gj = k + sb + jtr;
                        double acc = 0.0;
                        for (int gi = 0; gi < N; ++gi) {
                            double qv = h_panel_d[(size_t)gi + (size_t)t * N];
                            std::int64_t li, lj;
                            if (global_to_local_bc(gi, gj, b, Px, Py, my_px, my_py, li, lj))
                                acc += qv * h_loc_d[(size_t)li + (size_t)lj * locr];
                        }
                        h_W[(size_t)t + (size_t)jtr * sb] = acc;
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, h_W.data(), sb * ntrail, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                for (std::int64_t lj = 0; lj < locc; ++lj) {
                    for (std::int64_t li = 0; li < locr; ++li) {
                        int gi, gj;
                        local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                        if (gj < k + sb) continue;
                        int jtr = gj - k - sb;
                        double sum = 0.0;
                        for (int t = 0; t < sb; ++t)
                            sum += h_panel_d[(size_t)gi + (size_t)t * N] * h_W[(size_t)t + (size_t)jtr * sb];
                        h_loc_d[(size_t)li + (size_t)lj * locr] -= sum;
                    }
                }
                CUDA_CHECK(cudaMemcpy(d_loc_d, h_loc_d.data(), loc_sz * sizeof(double), cudaMemcpyHostToDevice));
            }
        }
    };

    auto reset = [&]() { seed_local(); };

    for (int i = 0; i < 2; ++i) {
        reset();
        run_qr_bc();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_hbc));
    MPI_Barrier(MPI_COMM_WORLD);

    const int nrun = 5;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qr_bc();
        CUDA_CHECK(cudaStreamSynchronize(stream_hbc));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        const char* matnm = matrix_mode_tag(A.matrix);
        printf("  %-30s  N=%d b=%d grid=[%d,%d,%d] layout=blockcyclic  tmin=%9.2f ms  tmed=%9.2f ms\n", matnm, N, b,
               Px, Py, A.pz, times[0], tmed);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, Px * Py);
        printf("METRICS bench=householder_2p5d matrix=%s layout=blockcyclic N=%d b=%d Px=%d Py=%d Pz=%d passes=1 ",
               matnm, N, b, Px, Py, 1);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    if (full_f) {
        cudaFree(d_loc_f);
        cudaFree(d_panel_f);
        cudaFree(d_tau_f);
        cudaFree(d_potwf);
    } else {
        cudaFree(d_loc_d);
        cudaFree(d_panel_d);
        cudaFree(d_tau);
        cudaFree(d_potw);
    }
    cudaFree(d_info);
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);
    cudaStreamDestroy(stream_hbc);
    destroy_p1_zslice_nccl(&p1);
    return 0;
}

#endif
