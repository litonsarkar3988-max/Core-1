#include <torch/torch.h>
#include <torch/distributed.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "config.h"

/*
=====================================================
  TITANCORE: ULTRA FSDP & PIPELINE MANAGER (1T+)
=====================================================
  Features:
  - ZeRO-3 Parameter Sharding (Scatter/Gather)
  - Reduce-Scatter Gradient Synchronization
  - Pipeline Stage Communication Hooks
=====================================================
*/

struct TitanFSDPManager {
    int rank, world_size;
    int pipeline_rank, pipeline_world_size;
    ncclComm_t comm;
    cudaStream_t stream;

    TitanFSDPManager(const TitanConfig& cfg, ncclComm_t c) : comm(c) {
        rank = torch::distributed::get_rank();
        world_size = torch::distributed::get_world_size();
        stream = at::cuda::getCurrentCUDAStream();
        
        // 1. Pipeline Stage Assignment
        pipeline_world_size = cfg.pipeline_parallel_size;
        int gpus_per_stage = world_size / pipeline_world_size;
        pipeline_rank = rank / gpus_per_stage;

        if (rank == 0) {
            std::cout << "[TitanCore] Ultra FSDP Initialized." << std::endl;
        }
    }

    

    // --- ZeRO-3: Parameter Sharding Logic ---

    /* Shard full weight and keep only local portion to save VRAM */
    torch::Tensor shard_weight(torch::Tensor full) {
        auto total = full.numel();
        auto shard_size = (total + world_size - 1) / world_size;
        
        // Allocate sharded buffer
        auto local_shard = torch::empty({shard_size}, full.options().device(torch::kCUDA));

        // NCCL Scatter: Full weight -> GPU Shards
        ncclScatter(
            full.data_ptr(),
            local_shard.data_ptr(),
            shard_size,
            ncclFloat16,
            0,
            comm,
            stream
        );

        return local_shard;
    }

    /* All-Gather: Collect shards from all GPUs before Forward Pass */
    void all_gather_parameters(torch::nn::ModuleList& layers, std::vector<torch::Tensor>& shards) {
        int shard_idx = 0;
        for (auto& layer : layers) {
            for (auto& p : layer->parameters()) {
                if (!p.defined()) continue;
                
                // NCCL AllGather: Shards -> Full Weights
                ncclAllGather(
                    shards[shard_idx].data_ptr(),
                    p.data_ptr(),
                    shards[shard_idx].numel(),
                    ncclFloat16,
                    comm,
                    stream
                );
                shard_idx++;
            }
        }
    }

    // --- ZeRO-2: Gradient Optimization ---

    /* Reduce-Scatter: Only store the gradient shard that matches the parameter shard */
    void reduce_scatter_gradients(torch::nn::ModuleList& layers) {
        for (auto& layer : layers) {
            for (auto& p : layer->parameters()) {
                if (p.grad().defined()) {
                    auto grad = p.grad();
                    auto shard_size = grad.numel() / world_size;

                    // Reduce-Scatter: Aggregate grads across GPUs but store only the 1/N portion
                    ncclReduceScatter(
                        grad.data_ptr(),
                        grad.data_ptr(), // In-place reduction to its local shard
                        shard_size,
                        ncclFloat16,
                        ncclSum,
                        comm,
                        stream
                    );
                    
                    // Division by world size for average
                    grad.div_(world_size);
                }
            }
        }
    }

    

    // --- Pipeline Communication (Placeholder for 3D Parallelism) ---

    void send_hidden_state(torch::Tensor x, int target_rank) {
        ncclSend(x.data_ptr(), x.numel(), ncclFloat16, target_rank, comm, stream);
    }

    void receive_hidden_state(torch::Tensor x, int source_rank) {
        ncclRecv(x.data_ptr(), x.numel(), ncclFloat16, source_rank, comm, stream);
    }
};
