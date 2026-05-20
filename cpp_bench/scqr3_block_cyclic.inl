// Included from scqr3_full25d_bench.cu after run_fp32_body().
// Pz=1 only: 2D block-cyclic (b×b) local [locr×locc] column-major (lda=locr).
// Panel: host gather + MPI_SUM, device SYRK/POTRF/TRSM, nccl_p1 AllReduce on G,
// scatter. Trailing: replicated Q (host), partial W per rank, MPI_SUM on W, local update.

#ifndef SCQR3_BLOCK_CYCLIC_INL
#define SCQR3_BLOCK_CYCLIC_INL

#include "nextla_mp_trail.hpp"

static int run_block_cyclic_scqr3_main(const Args& A, int P, int Px, int Py,
                                       int my_px, int my_py, int my_pz) {
    (void)P;
    int N = A.N, b = A.b;
    if (A.pz != 1) {
        if (_rank == 0) fprintf(stderr, "layout=blockcyclic: require Pz=1\n");
        return 2;
    }
    if (my_pz != 0) {
        if (_rank == 0) fprintf(stderr, "layout=blockcyclic: unexpected my_pz!=0 with Pz=1\n");
        return 2;
    }
    if (!block_cyclic_valid(N, b, b, Px, Py)) {
        if (_rank == 0) fprintf(stderr, "layout=blockcyclic: need N%%b==0, positive grid\n");
        return 3;
    }
    if (total_elements_bc(N, b, b, Px, Py) != (std::int64_t)N * N) {
        if (_rank == 0) fprintf(stderr, "layout=blockcyclic: element count mismatch\n");
        return 4;
    }

    std::int64_t locr = numroc(N, b, my_px, 0, Px);
    std::int64_t locc = numroc(N, b, my_py, 0, Py);

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    P1ZsliceNccl p1{};
    create_p1_zslice_nccl_full(MPI_COMM_WORLD, _rank, Px, Py, 1, my_px, my_py, my_pz, &p1);

    cudaStream_t stream_bc = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_bc));
    cublasHandle_t cublas{};
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream_bc));
    cusolverDnHandle_t cusolver{};
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream_bc));

    const bool full_f = (A.matrix == MatrixMode::FP32_FULL);
    const bool mp_trail = nextla_is_mp_trail_matrix(A.matrix);
    if (mp_trail && nextla_requests_tf32_matrix(A.matrix) && _rank == 0) {
        fprintf(stdout,
                "[note] scqr3 blockcyclic: fp64mp_tf32 uses host float accumulation on trailing W "
                "(same branch as fp64mp); slab path uses cuBLAS TF32 tensor ops where available.\n");
        fflush(stdout);
    }

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

    double* d_G = nullptr;
    float* d_Gf = nullptr;
    double* d_panel_d = nullptr;
    float* d_panel_f = nullptr;
    double* d_potw = nullptr;
    float* d_potwf = nullptr;
    int potrf_lwork = 0, potrf_lwork_f = 0;
    if (full_f) {
        CUDA_CHECK(cudaMalloc(&d_Gf, (size_t)b * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_panel_f, (size_t)N * b * sizeof(float)));
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_Gf, b, &potrf_lwork_f));
        CUDA_CHECK(cudaMalloc(&d_potwf, (size_t)potrf_lwork_f * sizeof(float)));
    } else {
        CUDA_CHECK(cudaMalloc(&d_G, (size_t)b * b * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_panel_d, (size_t)N * b * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
        CUDA_CHECK(cudaMalloc(&d_potw, (size_t)potrf_lwork * sizeof(double)));
    }
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

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
            // Gather panel columns k..k+sb-1
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
                MPI_Allreduce(MPI_IN_PLACE, h_panel_f.data(), (int)(N * sb), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                CUDA_CHECK(cudaMemcpy(d_panel_f, h_panel_f.data(), (size_t)N * sb * sizeof(float), cudaMemcpyHostToDevice));
                for (int it = 0; it < A.passes; ++it) {
                    CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, sb, N, &one_f, d_panel_f, N,
                                             &zero_f, d_Gf, b));
                    CUDA_CHECK(cudaStreamSynchronize(stream_bc));
                    NCCL_CHECK(ncclAllReduce(d_Gf, d_Gf, (size_t)b * b, ncclFloat, ncclSum, p1.nccl_p1, stream_bc));
                    CUDA_CHECK(cudaStreamSynchronize(stream_bc));
                    if (it == 0) {
                        float coef = 11.f * ((float)N * sb + (float)sb * (sb + 1)) * 1.1920929e-07f;
                        float* d_tr = nullptr;
                        CUDA_CHECK(cudaMalloc(&d_tr, sizeof(float)));
                        trace_b_kernel_f<<<1, std::min(sb, 1024), 0, stream_bc>>>(d_Gf, b, sb, d_tr);
                        shift_diag_from_trace_f_kernel<<<(sb + 255) / 256, 256, 0, stream_bc>>>(d_Gf, b, sb, d_tr, coef);
                        cudaFree(d_tr);
                    }
                    CUSOLVER_CHECK(cusolverDnSpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_Gf, b, d_potwf, potrf_lwork_f, d_info));
                    CUBLAS_CHECK(cublasStrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                             N, sb, &one_f, d_Gf, b, d_panel_f, N));
                }
                CUDA_CHECK(cudaMemcpy(h_panel_f.data(), d_panel_f, (size_t)N * sb * sizeof(float), cudaMemcpyDeviceToHost));
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
                MPI_Allreduce(MPI_IN_PLACE, h_panel_d.data(), (int)(N * sb), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                CUDA_CHECK(cudaMemcpy(d_panel_d, h_panel_d.data(), (size_t)N * sb * sizeof(double), cudaMemcpyHostToDevice));
                for (int it = 0; it < A.passes; ++it) {
                    CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, sb, N, &one_d, d_panel_d, N,
                                             &zero_d, d_G, b));
                    CUDA_CHECK(cudaStreamSynchronize(stream_bc));
                    NCCL_CHECK(ncclAllReduce(d_G, d_G, (size_t)b * b, ncclDouble, ncclSum, p1.nccl_p1, stream_bc));
                    CUDA_CHECK(cudaStreamSynchronize(stream_bc));
                    if (it == 0) {
                        double coef = 11.0 * ((double)N * sb + (double)sb * (sb + 1)) * 2.220446049250313e-16;
                        double* d_tr = nullptr;
                        CUDA_CHECK(cudaMalloc(&d_tr, sizeof(double)));
                        trace_b_kernel<<<1, std::min(sb, 1024), 0, stream_bc>>>(d_G, b, sb, d_tr);
                        shift_diag_from_trace_kernel<<<(sb + 255) / 256, 256, 0, stream_bc>>>(d_G, b, sb, d_tr, coef);
                        cudaFree(d_tr);
                    }
                    CUSOLVER_CHECK(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G, b, d_potw, potrf_lwork, d_info));
                    CUBLAS_CHECK(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                             N, sb, &one_d, d_G, b, d_panel_d, N));
                }
                CUDA_CHECK(cudaMemcpy(h_panel_d.data(), d_panel_d, (size_t)N * sb * sizeof(double), cudaMemcpyDeviceToHost));
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
                        for (int t = 0; t < sb; ++t)
                            sum += (double)h_Wf[(size_t)t + (size_t)jtr * sb] * h_panel_d[(size_t)gi + (size_t)t * N];
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

    auto reset = [&]() {
        seed_local();
    };

    for (int i = 0; i < 2; ++i) {
        reset();
        run_qr_bc();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_bc));
    MPI_Barrier(MPI_COMM_WORLD);

    const int nrun = 5;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qr_bc();
        CUDA_CHECK(cudaStreamSynchronize(stream_bc));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        const char* matnm = matrix_mode_tag(A.matrix);
        printf("  %-30s  N=%d b=%d grid=[%d,%d,%d] layout=blockcyclic  tmin=%9.2f ms  tmed=%9.2f ms\n",
               matnm, N, b, Px, Py, 1, times[0], tmed);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, Px * Py);
        printf("METRICS bench=scqr3_full25d matrix=%s layout=blockcyclic N=%d b=%d Px=%d Py=%d Pz=%d passes=%d ",
               matnm, N, b, Px, Py, 1, A.passes);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    if (full_f) {
        cudaFree(d_loc_f);
        cudaFree(d_Gf);
        cudaFree(d_panel_f);
        cudaFree(d_potwf);
    } else {
        cudaFree(d_loc_d);
        cudaFree(d_G);
        cudaFree(d_panel_d);
        cudaFree(d_potw);
    }
    cudaFree(d_info);
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);
    cudaStreamDestroy(stream_bc);
    destroy_p1_zslice_nccl(&p1);
    return 0;
}

#endif  // SCQR3_BLOCK_CYCLIC_INL
