// Path (q) QDWH with N×N block-cyclic storage for A and X (Pz=1), same π(I,J) as (h)/(g)/(s).
// Inner stacked QR on S = [sqrt(c)*X; I] is factorized on rank 0 after MPI_Allreduce-gather of S from BC
// tiles (dense replica on every rank), then Q is Bcast. Polar update (Q_1 Q_2^T blocks) runs on host on
// all ranks — duplicate O(N^3) but correct. Intended for correctness + METRICS at moderate N;
// distributed inner QR is future work.
//
// Matrix surface (match Path (s) / slab qdwh where math allows):
//   --matrix=fp64 | fp64mp | fp64mp_tf32 | fp32full
//   fp64: default inner = one-shot cusolverDnDgeqrf/Dorgqr on rank 0; use --la for panelized CQR2 + streams.
//   fp64mp / fp64mp_tf32: panelized CQR2 on rank 0 with float trailing GEMMs (TF32 when requested + toolkit).
//   fp32full: float BC tiles + float rank-0 panel inner; inner-QR --la N/A (main() clears default LA before BC).
//   --la / --no-la: honored for fp64 family (rank-0 pipelined inner); explicit --la on fp32full aborts (slab parity).

#ifndef QDWH_BLOCK_CYCLIC_INL
#define QDWH_BLOCK_CYCLIC_INL

#include "block_cyclic.hpp"
#include "comm_groups.cuh"
#include "bench_vendor_metrics.hpp"
#include "matrix_mode.hpp"
#include "nextla_mp_trail.hpp"

#include <chrono>
#include <random>

#include <cublas_v2.h>
#include <cusolverDn.h>

#if !defined(NEXTLA_HAVE_CUBLAS_TF32)
#if defined(CUBLAS_COMPUTE_32F_FAST_TF32)
#define NEXTLA_HAVE_CUBLAS_TF32 1
#else
#define NEXTLA_HAVE_CUBLAS_TF32 0
#endif
#endif

static int run_qdwh_bc_fp32full(const Args& A, int Px, int Py, int my_px, int my_py, int my_pz) {
    (void)my_pz;
    int N = A.N, b = A.b;
    if (!block_cyclic_valid(N, b, b, Px, Py)) {
        if (_rank == 0)
            fprintf(stderr, "qdwh blockcyclic: need N%%b==0 and positive grid\n");
        return 3;
    }
    if (A.strict_b && !b_in_window(b, 1, N, Px, Py)) {
        if (_rank == 0)
            fprintf(stderr, "qdwh blockcyclic: b=%d violates admissible window for N=%d pr=%d pc=%d\n", b, N, Px, Py);
        return 55;
    }

    std::int64_t locr = numroc(N, b, my_px, 0, Px);
    std::int64_t locc = numroc(N, b, my_py, 0, Py);
    std::int64_t loc_sz = std::max<std::int64_t>(1, locr) * std::max<std::int64_t>(1, locc);

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    nextla_maybe_print_tf32_trailing_banner_rank0(_rank, false);

    P1ZsliceNccl p1{};
    create_p1_zslice_nccl_full(MPI_COMM_WORLD, _rank, Px, Py, 1, my_px, my_py, my_pz, &p1);

    cudaStream_t stream_qbc = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_qbc));
    cublasHandle_t cublas{};
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream_qbc));
    cusolverDnHandle_t cusolver{};
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream_qbc));

    float* d_loc_A = nullptr;
    float* d_loc_X = nullptr;
    CUDA_CHECK(cudaMalloc(&d_loc_A, (size_t)loc_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loc_X, (size_t)loc_sz * sizeof(float)));

    auto seed_bc = [&]() {
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
        CUDA_CHECK(cudaMemcpy(d_loc_A, h.data(), loc_sz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_loc_X, d_loc_A, loc_sz * sizeof(float), cudaMemcpyDeviceToDevice));
    };
    seed_bc();

    std::vector<float> h_loc(locr * locc);
    std::vector<float> h_X((size_t)N * N, 0.f);
    std::vector<float> h_S((size_t)(2 * N) * N, 0.f);
    std::vector<float> h_Q((size_t)(2 * N) * N, 0.f);
    std::vector<float> h_P((size_t)N * N, 0.f);
    std::vector<float> h_Xnew((size_t)N * N, 0.f);
    const int ldS = 2 * N;

    auto dev_to_host_loc = [&](float* d) {
        CUDA_CHECK(cudaMemcpy(h_loc.data(), d, loc_sz * sizeof(float), cudaMemcpyDeviceToHost));
    };
    auto host_to_dev_loc = [&](float* d) {
        CUDA_CHECK(cudaMemcpy(d, h_loc.data(), loc_sz * sizeof(float), cudaMemcpyHostToDevice));
    };

    auto gather_dense_X = [&]() {
        std::fill(h_X.begin(), h_X.end(), 0.f);
        dev_to_host_loc(d_loc_X);
        for (std::int64_t lj = 0; lj < locc; ++lj) {
            for (std::int64_t li = 0; li < locr; ++li) {
                int gi, gj;
                local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                h_X[(size_t)gi + (size_t)gj * N] = h_loc[(size_t)li + (size_t)lj * locr];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, h_X.data(), N * N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    };

    auto scatter_dense_X = [&]() {
        for (std::int64_t lj = 0; lj < locc; ++lj) {
            for (std::int64_t li = 0; li < locr; ++li) {
                int gi, gj;
                local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                h_loc[(size_t)li + (size_t)lj * locr] = h_X[(size_t)gi + (size_t)gj * N];
            }
        }
        host_to_dev_loc(d_loc_X);
    };

    float* d_S0 = nullptr;
    float* d_G = nullptr;
    float* d_W = nullptr;
    float* d_potrf_work = nullptr;
    int* d_info = nullptr;
    int potrf_lwork = 0;
    if (_rank == 0) {
        CUDA_CHECK(cudaMalloc(&d_S0, (size_t)ldS * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_G, (size_t)b * b * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
        CUDA_CHECK(cudaMalloc(&d_potrf_work, (size_t)potrf_lwork * sizeof(float)));
    }

    const int m_st = ldS;
    const float one_f = 1.f, zero_f = 0.f, neg_one_f = -1.f;

    auto inner_qr_rank0_f = [&]() {
        if (_rank == 0) {
            CUDA_CHECK(cudaMemcpy(d_S0, h_S.data(), (size_t)ldS * N * sizeof(float), cudaMemcpyHostToDevice));
            int k = 0;
            while (k < N) {
                int sb = std::min(b, N - k);
                int n_tr = N - (k + sb);
                for (int it = 0; it < 2; ++it) {
                    float* d_S_panel = d_S0 + (size_t)k * m_st;
                    CUBLAS_CHECK(cublasSsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, sb, m_st, &one_f, d_S_panel,
                                             m_st, &zero_f, d_G, b));
                    CUSOLVER_CHECK(cusolverDnSpotrf(cusolver, CUBLAS_FILL_MODE_UPPER, sb, d_G, b, d_potrf_work,
                                                    potrf_lwork, d_info));
                    CUBLAS_CHECK(cublasStrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                             CUBLAS_DIAG_NON_UNIT, m_st, sb, &one_f, d_G, b, d_S_panel, m_st));
                }
                if (n_tr > 0) {
                    float* d_S_panel = d_S0 + (size_t)k * m_st;
                    float* d_S_tr = d_S0 + (size_t)(k + sb) * m_st;
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, sb, n_tr, m_st, &one_f, d_S_panel,
                                             m_st, d_S_tr, m_st, &zero_f, d_W, b));
                    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, m_st, n_tr, sb, &neg_one_f, d_S_panel,
                                             m_st, d_W, b, &one_f, d_S_tr, m_st));
                }
                k += sb;
            }
            CUDA_CHECK(cudaMemcpy(h_Q.data(), d_S0, (size_t)ldS * N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaStreamSynchronize(stream_qbc));
        }
        MPI_Bcast(h_Q.data(), ldS * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    };

    char tag[96];
    std::snprintf(tag, sizeof(tag), "qdwh_%s_bc_it=%d", matrix_mode_tag(A.matrix), A.iters);

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" QDWH 2.5D (block-cyclic fp32full)  N=%d b=%d grid=[%d,%d,1]  variant=%s\n", N, b, Px, Py, tag);
        printf("=================================================================\n");
        fflush(stdout);
    }

    auto run_qdwh_body = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(h_loc.data(), d_loc_A, loc_sz * sizeof(float), cudaMemcpyDeviceToHost));
        float frob_l = 0.f;
        for (std::int64_t lj = 0; lj < locc; ++lj) {
            for (std::int64_t li = 0; li < locr; ++li) {
                float v = h_loc[(size_t)li + (size_t)lj * locr];
                frob_l += v * v;
            }
        }
        float frob_g = 0.f;
        MPI_Allreduce(&frob_l, &frob_g, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        float alpha = std::sqrt(std::max(frob_g, 1e-30f));
        float inv_alpha = 1.f / alpha;
        gather_dense_X();
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i)
                h_X[(size_t)i + (size_t)j * N] *= inv_alpha;
        }
        scatter_dense_X();

        float l = 1.f / std::sqrt((float)N);
        if (l < 1e-15f)
            l = 1e-15f;

        for (int kit = 0; kit < A.iters; ++kit) {
            float l2 = l * l;
            float dd = std::pow(4.f * (1.f - l2) / (l2 * l2), 1.f / 3.f);
            float sd = std::sqrt(1.f + dd);
            float inner = std::max(0.f, 8.f - 4.f * dd + 8.f * (2.f - l2) / (l2 * sd));
            float a_k = sd + 0.5f * std::sqrt(inner);
            float b_k = (a_k - 1.f) * (a_k - 1.f) / 4.f;
            float c_k = a_k + b_k - 1.f;
            float scale_X = std::sqrt(c_k);

            gather_dense_X();
            for (int j = 0; j < N; ++j) {
                for (int gi = 0; gi < ldS; ++gi) {
                    if (gi < N)
                        h_S[(size_t)gi + (size_t)j * ldS] = scale_X * h_X[(size_t)gi + (size_t)j * N];
                    else
                        h_S[(size_t)gi + (size_t)j * ldS] = (((gi - N) == j) ? 1.f : 0.f);
                }
            }

            inner_qr_rank0_f();

            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.f;
                    for (int k = 0; k < N; ++k) {
                        float q1 = h_Q[(size_t)i + (size_t)k * ldS];
                        float q2 = h_Q[(size_t)(N + j) + (size_t)k * ldS];
                        sum += q1 * q2;
                    }
                    h_P[(size_t)i + (size_t)j * N] = sum;
                }
            }

            float alpha_x = b_k / c_k;
            float alpha_p = (a_k - b_k / c_k) / std::sqrt(c_k);
            gather_dense_X();
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    float xk = h_X[(size_t)i + (size_t)j * N];
                    float pv = h_P[(size_t)i + (size_t)j * N];
                    h_Xnew[(size_t)i + (size_t)j * N] = alpha_x * xk + alpha_p * pv;
                }
            }
            h_X.swap(h_Xnew);
            scatter_dense_X();

            float num = l * (a_k + b_k * l * l);
            float den = 1.f + c_k * l * l;
            l = num / den;
        }
        return l;
    };

    auto reset = [&]() {
        seed_bc();
        CUDA_CHECK(cudaMemcpy(d_loc_X, d_loc_A, loc_sz * sizeof(float), cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) {
        reset();
        run_qdwh_body();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_qbc));
    MPI_Barrier(MPI_COMM_WORLD);

    const int nrun = 3;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qdwh_body();
        CUDA_CHECK(cudaStreamSynchronize(stream_qbc));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        printf("  %-30s  N=%d b=%d grid=[%d,%d,1] layout=blockcyclic  tmin=%9.2f ms  tmed=%9.2f ms\n", tag, N, b, Px,
               Py, times[0], tmed);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, Px * Py);
        printf("METRICS bench=qdwh_2p5d matrix=fp32full layout=blockcyclic N=%d b=%d Px=%d Py=%d Pz=1 passes=1 ", N, b,
               Px, Py);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    if (_rank == 0) {
        cudaFree(d_S0);
        cudaFree(d_G);
        cudaFree(d_W);
        cudaFree(d_potrf_work);
        cudaFree(d_info);
    }
    cudaFree(d_loc_A);
    cudaFree(d_loc_X);
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);
    cudaStreamDestroy(stream_qbc);
    destroy_p1_zslice_nccl(&p1);
    return 0;
}

static int run_qdwh_bc_main(const Args& A, int P, int Px, int Py, int my_px, int my_py, int my_pz) {
    (void)P;
    if (A.pz != 1) {
        if (_rank == 0)
            fprintf(stderr, "qdwh blockcyclic: require Pz=1\n");
        return 61;
    }
    if (my_pz != 0) {
        if (_rank == 0)
            fprintf(stderr, "qdwh blockcyclic: unexpected my_pz!=0\n");
        return 61;
    }
    if (A.matrix == MatrixMode::FP32_FULL) {
        return run_qdwh_bc_fp32full(A, Px, Py, my_px, my_py, my_pz);
    }

    const bool use_mp_inner = nextla_is_mp_trail_matrix(A.matrix);
    const bool use_tf32_trail =
        nextla_requests_tf32_matrix(A.matrix) && (NEXTLA_HAVE_CUBLAS_TF32 != 0);
    if (nextla_requests_tf32_matrix(A.matrix) && !use_tf32_trail) {
        if (_rank == 0) {
            fprintf(stderr,
                    "qdwh blockcyclic: --matrix=fp64mp_tf32 requires CUDA 11+ cuBLAS (CUBLAS_COMPUTE_32F_FAST_TF32).\n");
        }
        return 91;
    }
    const bool geqrf_inner = (A.matrix == MatrixMode::FP64) && !A.lookahead && !use_mp_inner;

    int N = A.N, b = A.b;
    if (!block_cyclic_valid(N, b, b, Px, Py)) {
        if (_rank == 0)
            fprintf(stderr, "qdwh blockcyclic: need N%%b==0 and positive grid\n");
        return 3;
    }
    if (A.strict_b && !b_in_window(b, 1, N, Px, Py)) {
        if (_rank == 0)
            fprintf(stderr, "qdwh blockcyclic: b=%d violates admissible window for N=%d pr=%d pc=%d\n", b, N, Px, Py);
        return 55;
    }

    std::int64_t locr = numroc(N, b, my_px, 0, Px);
    std::int64_t locc = numroc(N, b, my_py, 0, Py);
    std::int64_t loc_sz = std::max<std::int64_t>(1, locr) * std::max<std::int64_t>(1, locc);

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(_rank % ngpu));

    nextla_maybe_print_tf32_trailing_banner_rank0(_rank, use_tf32_trail);
    if (_rank == 0 && nextla_requests_tf32_matrix(A.matrix)) {
        fprintf(stdout,
                "qdwh blockcyclic: rank-0 inner trailing uses cuBLAS TF32 where available (same intent as slab "
                "qdwh inner QR).\n");
        fflush(stdout);
    }

    P1ZsliceNccl p1{};
    create_p1_zslice_nccl_full(MPI_COMM_WORLD, _rank, Px, Py, 1, my_px, my_py, my_pz, &p1);

    cudaStream_t stream_qbc = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_qbc));
    cudaStream_t stream_la = nullptr;
    if (!geqrf_inner && A.lookahead && _rank == 0) {
        CUDA_CHECK(cudaStreamCreate(&stream_la));
    }
    cublasHandle_t cublas{};
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream_qbc));
    cublasHandle_t cublas_la{};
    if (!geqrf_inner && A.lookahead && _rank == 0) {
        CUBLAS_CHECK(cublasCreate(&cublas_la));
        CUBLAS_CHECK(cublasSetStream(cublas_la, stream_la));
    }
    cusolverDnHandle_t cusolver{};
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream_qbc));
    cusolverDnHandle_t cusolver_la{};
    if (!geqrf_inner && A.lookahead && _rank == 0) {
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_la));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver_la, stream_la));
    }

    cudaEvent_t e_comp_done = nullptr, e_ar_done = nullptr, e_panel_done = nullptr, e_next_ready = nullptr;
    if (!geqrf_inner && A.lookahead && _rank == 0) {
        CUDA_CHECK(cudaEventCreateWithFlags(&e_comp_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&e_ar_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&e_panel_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&e_next_ready, cudaEventDisableTiming));
    }

    double* d_loc_A = nullptr;
    double* d_loc_X = nullptr;
    CUDA_CHECK(cudaMalloc(&d_loc_A, (size_t)loc_sz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_loc_X, (size_t)loc_sz * sizeof(double)));

    auto seed_bc = [&]() {
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
        CUDA_CHECK(cudaMemcpy(d_loc_A, h.data(), loc_sz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_loc_X, d_loc_A, loc_sz * sizeof(double), cudaMemcpyDeviceToDevice));
    };
    seed_bc();

    std::vector<double> h_loc(locr * locc);
    std::vector<double> h_X((size_t)N * N, 0.0);
    std::vector<double> h_S((size_t)(2 * N) * N, 0.0);
    std::vector<double> h_Q((size_t)(2 * N) * N, 0.0);
    std::vector<double> h_P((size_t)N * N, 0.0);
    std::vector<double> h_Xnew((size_t)N * N, 0.0);
    const int ldS = 2 * N;
    const int m_st = ldS;

    auto dev_to_host_loc = [&](double* d) {
        CUDA_CHECK(cudaMemcpy(h_loc.data(), d, loc_sz * sizeof(double), cudaMemcpyDeviceToHost));
    };
    auto host_to_dev_loc = [&](double* d) {
        CUDA_CHECK(cudaMemcpy(d, h_loc.data(), loc_sz * sizeof(double), cudaMemcpyHostToDevice));
    };

    auto gather_dense_X = [&]() {
        std::fill(h_X.begin(), h_X.end(), 0.0);
        dev_to_host_loc(d_loc_X);
        for (std::int64_t lj = 0; lj < locc; ++lj) {
            for (std::int64_t li = 0; li < locr; ++li) {
                int gi, gj;
                local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                h_X[(size_t)gi + (size_t)gj * N] = h_loc[(size_t)li + (size_t)lj * locr];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, h_X.data(), N * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    };

    auto scatter_dense_X = [&]() {
        for (std::int64_t lj = 0; lj < locc; ++lj) {
            for (std::int64_t li = 0; li < locr; ++li) {
                int gi, gj;
                local_to_global_bc(li, lj, b, Px, Py, my_px, my_py, gi, gj);
                h_loc[(size_t)li + (size_t)lj * locr] = h_X[(size_t)gi + (size_t)gj * N];
            }
        }
        host_to_dev_loc(d_loc_X);
    };

    double* d_S0 = nullptr;
    double* d_tau0 = nullptr;
    double* d_work0 = nullptr;
    int* d_info0 = nullptr;
    int lwork0 = 0;

    double* d_G = nullptr;
    double* d_W = nullptr;
    double* d_potrf_work = nullptr;
    int* d_info = nullptr;
    int potrf_lwork = 0;
    double* d_G2 = nullptr;
    double* d_potrf_work2 = nullptr;
    int* d_info2 = nullptr;
    float* d_Wf = nullptr;
    float* d_pf_panel = nullptr;
    float* d_pf_tr = nullptr;

    if (_rank == 0) {
        CUDA_CHECK(cudaMalloc(&d_S0, (size_t)ldS * N * sizeof(double)));
        if (geqrf_inner) {
            CUDA_CHECK(cudaMalloc(&d_tau0, (size_t)N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_info0, sizeof(int)));
            CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolver, ldS, N, d_S0, ldS, &lwork0));
            int lwo = 0;
            CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolver, ldS, N, N, d_S0, ldS, d_tau0, &lwo));
            lwork0 = std::max(lwork0, lwo);
            CUDA_CHECK(cudaMalloc(&d_work0, (size_t)lwork0 * sizeof(double)));
        } else {
            CUDA_CHECK(cudaMalloc(&d_G, (size_t)b * b * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_W, (size_t)b * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
            CUSOLVER_CHECK(
                cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_UPPER, b, d_G, b, &potrf_lwork));
            CUDA_CHECK(cudaMalloc(&d_potrf_work, (size_t)potrf_lwork * sizeof(double)));
            if (A.lookahead) {
                CUDA_CHECK(cudaMalloc(&d_G2, (size_t)b * b * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_potrf_work2, (size_t)potrf_lwork * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_info2, sizeof(int)));
            }
            if (use_mp_inner) {
                CUDA_CHECK(cudaMalloc(&d_Wf, (size_t)b * N * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_pf_panel, (size_t)m_st * b * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_pf_tr, (size_t)m_st * N * sizeof(float)));
            }
        }
    }

    const double one_d = 1.0, zero_d = 0.0, neg_one_d = -1.0;
    const float one_f = 1.f, zero_f = 0.f;

    auto bc_phase_q1 = [&](cudaStream_t s_use, cublasHandle_t cb, cusolverDnHandle_t cs, double* Gb, double* work,
                           int* info, int k, int sb) {
        for (int it = 0; it < 2; ++it) {
            double* d_S_panel = d_S0 + (size_t)k * m_st;
            CUBLAS_CHECK(cublasDsyrk(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, sb, m_st, &one_d, d_S_panel, m_st,
                                     &zero_d, Gb, b));
            CUSOLVER_CHECK(cusolverDnDpotrf(cs, CUBLAS_FILL_MODE_UPPER, sb, Gb, b, work, potrf_lwork, info));
            CUBLAS_CHECK(cublasDtrsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                     m_st, sb, &one_d, Gb, b, d_S_panel, m_st));
        }
    };

    auto bc_phase_q2 = [&](cudaStream_t s_use, cublasHandle_t cb, int k, int sb, int col_start, int ncols) {
        double* d_S_panel = d_S0 + (size_t)k * m_st;
        double* d_S_tr = d_S0 + (size_t)col_start * m_st;
        if (!use_mp_inner) {
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, sb, ncols, m_st, &one_d, d_S_panel, m_st, d_S_tr,
                                     m_st, &zero_d, d_W, b));
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, m_st, ncols, sb, &neg_one_d, d_S_panel, m_st, d_W, b,
                                     &one_d, d_S_tr, m_st));
        } else {
            const size_t np = (size_t)m_st * (size_t)sb;
            const size_t nt = (size_t)m_st * (size_t)ncols;
            const int ntiles = 256;
            cast_d2f_q<<<(unsigned)((np + ntiles - 1) / ntiles), ntiles, 0, s_use>>>(d_S_panel, d_pf_panel, np);
            cast_d2f_q<<<(unsigned)((nt + ntiles - 1) / ntiles), ntiles, 0, s_use>>>(d_S_tr, d_pf_tr, nt);
            if (use_tf32_trail) {
#if NEXTLA_HAVE_CUBLAS_TF32
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, sb, ncols, m_st, &one_f, d_pf_panel,
                                         CUDA_R_32F, m_st, d_pf_tr, CUDA_R_32F, m_st, &zero_f, d_Wf, CUDA_R_32F, b,
                                         CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, sb, ncols, m_st, &one_f, d_pf_panel, m_st,
                                         d_pf_tr, m_st, &zero_f, d_Wf, b));
#endif
            } else {
                CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, sb, ncols, m_st, &one_f, d_pf_panel, m_st,
                                         d_pf_tr, m_st, &zero_f, d_Wf, b));
            }
            cast_f2d_q<<<(unsigned)(((size_t)sb * (size_t)ncols + ntiles - 1) / ntiles), ntiles, 0, s_use>>>(
                d_Wf, d_W, (size_t)sb * (size_t)ncols);
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, m_st, ncols, sb, &neg_one_d, d_S_panel, m_st, d_W, b,
                                     &one_d, d_S_tr, m_st));
        }
    };

    auto inner_qr_rank0_panel = [&]() {
        if (_rank != 0)
            return;
        int k = 0;
        if (!A.lookahead) {
            while (k < N) {
                int sb = std::min(b, N - k);
                int n_tr = N - (k + sb);
                bc_phase_q1(stream_qbc, cublas, cusolver, d_G, d_potrf_work, d_info, k, sb);
                if (n_tr > 0)
                    bc_phase_q2(stream_qbc, cublas, k, sb, k + sb, n_tr);
                k += sb;
            }
        } else {
            int sb = std::min(b, N - k);
            bc_phase_q1(stream_qbc, cublas, cusolver, d_G, d_potrf_work, d_info, k, sb);
            CUDA_CHECK(cudaEventRecord(e_panel_done, stream_qbc));
            while (k < N) {
                int next_k = k + sb;
                int next_sb = (next_k < N) ? std::min(b, N - next_k) : 0;
                int rest_start = next_k + next_sb;
                int n_rest = (rest_start < N) ? (N - rest_start) : 0;
                if (next_sb > 0) {
                    bc_phase_q2(stream_qbc, cublas, k, sb, next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_next_ready, stream_qbc));
                    CUDA_CHECK(cudaStreamWaitEvent(stream_la, e_next_ready, 0));
                    bc_phase_q1(stream_la, cublas_la, cusolver_la, d_G2, d_potrf_work2, d_info2, next_k, next_sb);
                    CUDA_CHECK(cudaEventRecord(e_panel_done, stream_la));
                }
                if (n_rest > 0)
                    bc_phase_q2(stream_qbc, cublas, k, sb, rest_start, n_rest);
                CUDA_CHECK(cudaStreamWaitEvent(stream_qbc, e_panel_done, 0));
                k = next_k;
                sb = next_sb;
                if (sb == 0)
                    break;
            }
        }
        CUDA_CHECK(cudaMemcpy(h_Q.data(), d_S0, (size_t)ldS * N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(stream_qbc));
        if (A.lookahead)
            CUDA_CHECK(cudaStreamSynchronize(stream_la));
    };

    auto inner_qr_dense_rank0_geqrf = [&]() {
        if (_rank == 0) {
            CUSOLVER_CHECK(cusolverDnDgeqrf(cusolver, ldS, N, d_S0, ldS, d_tau0, d_work0, lwork0, d_info0));
            CUSOLVER_CHECK(cusolverDnDorgqr(cusolver, ldS, N, N, d_S0, ldS, d_tau0, d_work0, lwork0, d_info0));
            CUDA_CHECK(cudaMemcpy(h_Q.data(), d_S0, (size_t)ldS * N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaStreamSynchronize(stream_qbc));
        }
    };

    auto inner_qr_dispatch = [&]() {
        if (_rank == 0)
            CUDA_CHECK(cudaMemcpy(d_S0, h_S.data(), (size_t)ldS * N * sizeof(double), cudaMemcpyHostToDevice));
        if (geqrf_inner) {
            inner_qr_dense_rank0_geqrf();
        } else {
            inner_qr_rank0_panel();
        }
        MPI_Bcast(h_Q.data(), ldS * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    };

    char tag[160];
    std::snprintf(tag, sizeof(tag), "qdwh_%s%s_bc_it=%d", matrix_mode_tag(A.matrix), A.lookahead ? "+LA" : "",
                  A.iters);

    if (_rank == 0) {
        printf("=================================================================\n");
        printf(" QDWH 2.5D (block-cyclic)  N=%d b=%d grid=[%d,%d,1]  variant=%s\n", N, b, Px, Py, tag);
        printf("=================================================================\n");
        fflush(stdout);
    }

    auto run_qdwh_body = [&]() -> double {
        CUDA_CHECK(cudaMemcpy(h_loc.data(), d_loc_A, loc_sz * sizeof(double), cudaMemcpyDeviceToHost));
        double frob_l = 0.0;
        for (std::int64_t lj = 0; lj < locc; ++lj) {
            for (std::int64_t li = 0; li < locr; ++li) {
                double v = h_loc[(size_t)li + (size_t)lj * locr];
                frob_l += v * v;
            }
        }
        double frob_g = 0.0;
        MPI_Allreduce(&frob_l, &frob_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = std::sqrt(frob_g);
        if (alpha <= 0)
            alpha = 1.0;
        double inv_alpha = 1.0 / alpha;
        gather_dense_X();
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i)
                h_X[(size_t)i + (size_t)j * N] *= inv_alpha;
        }
        scatter_dense_X();

        double l = 1.0 / std::sqrt((double)N);
        if (l < 1e-15)
            l = 1e-15;

        for (int kit = 0; kit < A.iters; ++kit) {
            double l2 = l * l;
            double dd = std::pow(4.0 * (1.0 - l2) / (l2 * l2), 1.0 / 3.0);
            double sd = std::sqrt(1.0 + dd);
            double inner = std::max(0.0, 8.0 - 4.0 * dd + 8.0 * (2.0 - l2) / (l2 * sd));
            double a_k = sd + 0.5 * std::sqrt(inner);
            double b_k = (a_k - 1.0) * (a_k - 1.0) / 4.0;
            double c_k = a_k + b_k - 1.0;
            double scale_X = std::sqrt(c_k);

            gather_dense_X();
            for (int j = 0; j < N; ++j) {
                for (int gi = 0; gi < ldS; ++gi) {
                    if (gi < N)
                        h_S[(size_t)gi + (size_t)j * ldS] = scale_X * h_X[(size_t)gi + (size_t)j * N];
                    else
                        h_S[(size_t)gi + (size_t)j * ldS] = ((gi - N) == j) ? 1.0 : 0.0;
                }
            }

            inner_qr_dispatch();

            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    double sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        double q1 = h_Q[(size_t)i + (size_t)k * ldS];
                        double q2 = h_Q[(size_t)(N + j) + (size_t)k * ldS];
                        sum += q1 * q2;
                    }
                    h_P[(size_t)i + (size_t)j * N] = sum;
                }
            }

            double alpha_x = b_k / c_k;
            double alpha_p = (a_k - b_k / c_k) / std::sqrt(c_k);
            gather_dense_X();
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    double xk = h_X[(size_t)i + (size_t)j * N];
                    double pv = h_P[(size_t)i + (size_t)j * N];
                    h_Xnew[(size_t)i + (size_t)j * N] = alpha_x * xk + alpha_p * pv;
                }
            }
            h_X.swap(h_Xnew);
            scatter_dense_X();

            double num = l * (a_k + b_k * l * l);
            double den = 1.0 + c_k * l * l;
            l = num / den;
        }
        return l;
    };

    auto reset = [&]() {
        seed_bc();
        CUDA_CHECK(cudaMemcpy(d_loc_X, d_loc_A, loc_sz * sizeof(double), cudaMemcpyDeviceToDevice));
    };

    for (int i = 0; i < 2; ++i) {
        reset();
        run_qdwh_body();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_qbc));
    MPI_Barrier(MPI_COMM_WORLD);

    const int nrun = 3;
    std::vector<double> times(nrun);
    for (int i = 0; i < nrun; ++i) {
        reset();
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_qdwh_body();
        CUDA_CHECK(cudaStreamSynchronize(stream_qbc));
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    if (_rank == 0) {
        double tmed = times[nrun / 2];
        printf("  %-30s  N=%d b=%d grid=[%d,%d,1] layout=blockcyclic  tmin=%9.2f ms  tmed=%9.2f ms\n", tag, N, b, Px,
               Py, times[0], tmed);
        NextlaVendorMs vms = nextla_read_vendor_ms_for_np(N, Px * Py);
        printf("METRICS bench=qdwh_2p5d matrix=%s layout=blockcyclic N=%d b=%d Px=%d Py=%d Pz=1 passes=1 ",
               matrix_mode_tag(A.matrix), N, b, Px, Py);
        nextla_fprint_metrics_vendor_columns(stdout, vms);
        printf(" ours_ms=%.4f\n", tmed);
        fflush(stdout);
    }

    if (_rank == 0) {
        cudaFree(d_S0);
        if (geqrf_inner) {
            cudaFree(d_tau0);
            cudaFree(d_work0);
            cudaFree(d_info0);
        } else {
            cudaFree(d_G);
            cudaFree(d_W);
            cudaFree(d_potrf_work);
            cudaFree(d_info);
            if (A.lookahead) {
                cudaFree(d_G2);
                cudaFree(d_potrf_work2);
                cudaFree(d_info2);
            }
            if (use_mp_inner) {
                cudaFree(d_Wf);
                cudaFree(d_pf_panel);
                cudaFree(d_pf_tr);
            }
        }
    }
    cudaFree(d_loc_A);
    cudaFree(d_loc_X);
    if (!geqrf_inner && A.lookahead && _rank == 0) {
        cublasDestroy(cublas_la);
        cusolverDnDestroy(cusolver_la);
        cudaEventDestroy(e_comp_done);
        cudaEventDestroy(e_ar_done);
        cudaEventDestroy(e_panel_done);
        cudaEventDestroy(e_next_ready);
        cudaStreamDestroy(stream_la);
    }
    cublasDestroy(cublas);
    cusolverDnDestroy(cusolver);
    cudaStreamDestroy(stream_qbc);
    destroy_p1_zslice_nccl(&p1);
    return 0;
}

#endif
