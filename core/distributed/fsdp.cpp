#include <torch/torch.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>

/*
========================================
 TITANCORE FSDP ENGINE
 Fully Sharded Data Parallel
========================================
*/

struct TitanFSDP {

    int rank;
    int world;
    ncclComm_t comm;

    TitanFSDP(int r, int w, ncclComm_t c)
        : rank(r), world(w), comm(c) {}

    /* shard parameter tensor */

    torch::Tensor shard(torch::Tensor full) {

        auto total = full.numel();
        auto shard_size = total / world;

        auto local = torch::zeros({shard_size}, full.options());

        ncclScatter(
            full.data_ptr(),
            local.data_ptr(),
            shard_size,
            ncclFloat,
            0,
            comm,
            at::cuda::getDefaultCUDAStream()
        );

        return local;
    }

    /* gather full weight before forward */

    torch::Tensor gather(torch::Tensor local) {

        auto full = torch::zeros({local.numel() * world}, local.options());

        ncclAllGather(
            local.data_ptr(),
            full.data_ptr(),
            local.numel(),
            ncclFloat,
            comm,
            at::cuda::getDefaultCUDAStream()
        );

        return full;
    }

    /* reduce gradients */

    void reduce_scatter(torch::Tensor grad) {

        auto shard = torch::zeros_like(grad.slice(0, 0, grad.numel() / world));

        ncclReduceScatter(
            grad.data_ptr(),
            shard.data_ptr(),
            shard.numel(),
            ncclFloat,
            ncclSum,
            comm,
            at::cuda::getDefaultCUDAStream()
        );
    }
};
