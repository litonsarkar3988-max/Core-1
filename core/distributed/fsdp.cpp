#include <torch/torch.h>
#include <torch/distributed.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "config.h"

/*
=====================================================
  TITANCORE: ADVANCED FSDP & PIPELINE MANAGER
=====================================================
  Handles parameter sharding and gradient
  synchronization across GPUs using NCCL.
=====================================================
*/

struct TitanFSDPManager {
    int rank, world_size;
    int pipeline_rank, pipeline_world_size;
    ncclComm_t comm;

    TitanFSDPManager(const TitanConfig& cfg, ncclComm_t c) : comm(c) {
        // NCCL গ্রুপ থেকে র‍্যাঙ্ক এবং ওয়ার্ল্ড সাইজ নেওয়া
        rank = torch::distributed::get_rank();
        world_size = torch::distributed::get_world_size();
        
        // 1. Pipeline Parallelism Calculation
        pipeline_world_size = cfg.pipeline_parallel_size; // YAML: 2
        int gpus_per_stage = world_size / pipeline_world_size;
        
        // প্রতিটি GPU কোন পাইপলাইন স্টেজে আছে তা নির্ধারণ করা
        pipeline_rank = rank / gpus_per_stage;

        std::cout << "[TitanCore] FSDP Initialized. Rank: " << rank 
                  << " | Pipeline Stage: " << pipeline_rank << std::endl;
    }

    // 
    
    /* shard parameter tensor (Enhanced to support Sharded Weights) */
    torch::Tensor shard_weight(torch::Tensor full) {
        auto total = full.numel();
        
        // নিশ্চিত করা যে টেনসরটি ওয়ার্ল্ড সাইজ দিয়ে বিভাজ্য
        if (total % world_size != 0) {
            throw std::runtime_error("Tensor size must be divisible by world size for FSDP.");
        }
        
        auto shard_size = total / world_size; 
        auto local = torch::zeros({shard_size}, full.options().device(torch::kCUDA));

        // NCCL Scatter ব্যবহার করে পুরো মডেলের ওয়েট শার্ড করা
        ncclScatter(
            full.data_ptr(),
            local.data_ptr(),
            shard_size,
            ncclFloat16, // Ultra-scale-এর জন্য Float16 বা BFloat16 ব্যবহার করা উচিত
            0,
            comm,
            at::cuda::getCurrentCUDAStream()
        );

        return local;
    }

    /* gather full weight before forward pass */
    torch::Tensor gather_weight(torch::Tensor local) {
        auto full = torch::zeros({local.numel() * world_size}, local.options().device(torch::kCUDA));

        // NCCL AllGather ব্যবহার করে সব শার্ড একত্রিত করা
        ncclAllGather(
            local.data_ptr(),
            full.data_ptr(),
            local.numel(),
            ncclFloat16,
            comm,
            at::cuda::getCurrentCUDAStream()
        );

        return full;
    }

    // 
    
    /* Synchronize gradients across all GPUs */
    void synchronize_gradients(torch::nn::ModuleList& model_layers) {
        for (auto& layer : model_layers) {
            for (auto& param : layer->parameters()) {
                if (param.grad().defined()) {
                    // All-Reduce gradients to synchronize them
                    ncclAllReduce(
                        param.grad().data_ptr(),
                        param.grad().data_ptr(),
                        param.grad().numel(),
                        ncclFloat16,
                        ncclSum,
                        comm,
                        at::cuda::getCurrentCUDAStream()
                    );
                    // Average the gradients by dividing by world size
                    param.grad().div_(world_size);
                }
            }
        }
    }
};
