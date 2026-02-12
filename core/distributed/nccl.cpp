#include <torch/torch.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>

/*
========================================
 TITANCORE NCCL DISTRIBUTED BACKEND
 Tensor / Data Parallel Infrastructure
========================================
*/

struct TitanDistributed {

    int rank = 0;
    int world = 1;
    ncclComm_t comm;

    TitanDistributed(int r, int w, ncclUniqueId id) : rank(r), world(w) {
        ncclCommInitRank(&comm, world, id, rank);
        cudaSetDevice(rank);
        std::cout << "[NCCL] Rank " << rank << " initialized\n";
    }

    /* All Reduce */

    void all_reduce(torch::Tensor t) {

        ncclAllReduce(
            t.data_ptr(),
            t.data_ptr(),
            t.numel(),
            ncclFloat,
            ncclSum,
            comm,
            at::cuda::getDefaultCUDAStream()
        );
    }

    /* Broadcast from rank 0 */

    void broadcast(torch::Tensor t) {

        ncclBroadcast(
            t.data_ptr(),
            t.data_ptr(),
            t.numel(),
            ncclFloat,
            0,
            comm,
            at::cuda::getDefaultCUDAStream()
        );
    }

    ~TitanDistributed() {
        ncclCommDestroy(comm);
    }
};
