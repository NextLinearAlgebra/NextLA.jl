// comm_groups.cuh — Reusable MPI_Comm_split + NCCL pair for Conflux-style
// [px,py,pz] grids used by scqr3_full25d_bench.cu (and future shared layouts).
//
// Rank linearisation must match the bench: rank = pz*(px*py) + px*py + py
//   (equivalently: my_pz = rank/(px*py), my_px = (rank/py)%px, my_py = rank%py).

#ifndef COMM_GROUPS_CUH
#define COMM_GROUPS_CUH

#include <mpi.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>

struct ColRowNccl {
    MPI_Comm mpi_col_comm = MPI_COMM_NULL;
    MPI_Comm mpi_row_comm = MPI_COMM_NULL;
    ncclComm_t nccl_col = nullptr;
    ncclComm_t nccl_row = nullptr;
};

// All ranks on the same z-slice share this communicator (size Px*Py).
struct P1ZsliceNccl {
    MPI_Comm mpi_p1 = MPI_COMM_NULL;
    ncclComm_t nccl_p1 = nullptr;
};

inline void create_mpi_p1_zslice(MPI_Comm world, int my_pz, int my_px, int my_py, int Px, int Py,
                                  MPI_Comm* mpi_p1_out) {
    int color = my_pz;
    int key = my_px * Py + my_py;
    MPI_Comm_split(world, color, key, mpi_p1_out);
}

inline void init_nccl_p1(MPI_Comm mpi_p1, ncclComm_t* nccl_p1, int rank_for_abort) {
    int sz, rk;
    MPI_Comm_size(mpi_p1, &sz);
    MPI_Comm_rank(mpi_p1, &rk);
    ncclUniqueId id;
    if (rk == 0) {
        ncclResult_t r = ncclGetUniqueId(&id);
        if (r != ncclSuccess) {
            fprintf(stderr, "[r%d] ncclGetUniqueId(p1) %s\n", rank_for_abort, ncclGetErrorString(r));
            MPI_Abort(MPI_COMM_WORLD, 44);
        }
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_p1);
    ncclResult_t r = ncclCommInitRank(nccl_p1, sz, id, rk);
    if (r != ncclSuccess) {
        fprintf(stderr, "[r%d] ncclCommInitRank(p1) %s\n", rank_for_abort, ncclGetErrorString(r));
        MPI_Abort(MPI_COMM_WORLD, 45);
    }
}

inline void create_p1_zslice_nccl_full(MPI_Comm world, int rank,
                                       int Px, int Py, int Pz, int my_px, int my_py, int my_pz,
                                       P1ZsliceNccl* out) {
    create_mpi_p1_zslice(world, my_pz, my_px, my_py, Px, Py, &out->mpi_p1);
    init_nccl_p1(out->mpi_p1, &out->nccl_p1, rank);
}

inline void destroy_p1_zslice_nccl(P1ZsliceNccl* p) {
    if (p->nccl_p1) { ncclCommDestroy(p->nccl_p1); p->nccl_p1 = nullptr; }
    if (p->mpi_p1 != MPI_COMM_NULL) { MPI_Comm_free(&p->mpi_p1); }
}

// col_comm: fixed my_py (AllReduce G/W over px×pz).  row_comm: fixed (px,pz) (bcast Q over py).
inline void create_mpi_col_row_comms(MPI_Comm world, int rank,
                                     int Px, int Py, int Pz,
                                     int my_px, int my_py, int my_pz,
                                     MPI_Comm* mpi_col_comm, MPI_Comm* mpi_row_comm) {
    int col_color = my_py;
    int col_key = my_pz * Px + my_px;
    MPI_Comm_split(world, col_color, col_key, mpi_col_comm);
    int row_color = my_pz * Px + my_px;
    int row_key = my_py;
    MPI_Comm_split(world, row_color, row_key, mpi_row_comm);
}

inline void init_nccl_for_mpi_subcomms(MPI_Comm mpi_col_comm, MPI_Comm mpi_row_comm,
                                       ncclComm_t* nccl_col, ncclComm_t* nccl_row,
                                       int rank_for_abort) {
    int col_size, col_rank, row_size, row_rank;
    MPI_Comm_size(mpi_col_comm, &col_size);
    MPI_Comm_rank(mpi_col_comm, &col_rank);
    MPI_Comm_size(mpi_row_comm, &row_size);
    MPI_Comm_rank(mpi_row_comm, &row_rank);

    {
        ncclUniqueId id;
        if (col_rank == 0) {
            ncclResult_t r = ncclGetUniqueId(&id);
            if (r != ncclSuccess) {
                fprintf(stderr, "[r%d] ncclGetUniqueId(col) %s\n", rank_for_abort, ncclGetErrorString(r));
                MPI_Abort(MPI_COMM_WORLD, 40);
            }
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_col_comm);
        ncclResult_t r = ncclCommInitRank(nccl_col, col_size, id, col_rank);
        if (r != ncclSuccess) {
            fprintf(stderr, "[r%d] ncclCommInitRank(col) %s\n", rank_for_abort, ncclGetErrorString(r));
            MPI_Abort(MPI_COMM_WORLD, 41);
        }
    }
    {
        ncclUniqueId id;
        if (row_rank == 0) {
            ncclResult_t r = ncclGetUniqueId(&id);
            if (r != ncclSuccess) {
                fprintf(stderr, "[r%d] ncclGetUniqueId(row) %s\n", rank_for_abort, ncclGetErrorString(r));
                MPI_Abort(MPI_COMM_WORLD, 42);
            }
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_row_comm);
        ncclResult_t r = ncclCommInitRank(nccl_row, row_size, id, row_rank);
        if (r != ncclSuccess) {
            fprintf(stderr, "[r%d] ncclCommInitRank(row) %s\n", rank_for_abort, ncclGetErrorString(r));
            MPI_Abort(MPI_COMM_WORLD, 43);
        }
    }
}

inline void create_col_row_nccl_full(MPI_Comm world, int rank,
                                     int Px, int Py, int Pz,
                                     int my_px, int my_py, int my_pz,
                                     ColRowNccl* out) {
    create_mpi_col_row_comms(world, rank, Px, Py, Pz, my_px, my_py, my_pz,
                             &out->mpi_col_comm, &out->mpi_row_comm);
    init_nccl_for_mpi_subcomms(out->mpi_col_comm, out->mpi_row_comm,
                               &out->nccl_col, &out->nccl_row, rank);
}

inline void destroy_col_row_nccl(ColRowNccl* cr) {
    if (cr->nccl_col) { ncclCommDestroy(cr->nccl_col); cr->nccl_col = nullptr; }
    if (cr->nccl_row) { ncclCommDestroy(cr->nccl_row); cr->nccl_row = nullptr; }
    if (cr->mpi_col_comm != MPI_COMM_NULL) { MPI_Comm_free(&cr->mpi_col_comm); }
    if (cr->mpi_row_comm != MPI_COMM_NULL) { MPI_Comm_free(&cr->mpi_row_comm); }
}

#endif  // COMM_GROUPS_CUH
