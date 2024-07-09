/*
 * nvcc -g -I/opt/nccl/include -I/opt/openmpi/include -L/opt/openmpi/lib -L/opt/nccl/lib nccltest.c -o nccltest -lmpi -lnccl
 *
 * NCCL_DEBUG=info mpirun --np 2 ./nccltest (2 GPUs in the host)
 *
 */

#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaSetDevice(rank);
    if (ncclCommInitRank(&comm, size, id, rank) != ncclSuccess) {
        printf("ncclCommInitRank failed.\n");
    }

    // Perform NCCL operations...

    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);

    MPI_Finalize();
    return 0;
}
