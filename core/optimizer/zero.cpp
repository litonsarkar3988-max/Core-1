#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <memory>
#include "../../distributed/nccl.cpp"
#include "../../distributed/fsdp.cpp"
#include "../model/config.h"

/*
=====================================================
  TITANCORE: ZeRO-3 ULTRA OPTIMIZER (1T+ SCALING)
=====================================================
  Description: 
  - Stage 1: Shards Optimizer States (12x memory reduction)
  - Stage 2: Shards Gradients (Eliminates redundancy)
  - Stage 3: Shards Parameters (Enables 1T+ models)
=====================================================
*/

class TitanZeRO3 {
private:
    // Optimizer core
    std::unique_ptr<torch::optim::AdamW> optimizer;
    
    // Sharding references
    TitanFSDPManager* fsdp_manager;
    std::vector<torch::Tensor> local_param_shards;
    
    // Hyperparameters
    float lr;
    float weight_decay;
    float eps;

    // Distributed Info
    int rank;
    int world_size;

public:
    TitanZeRO3(const TitanConfig& cfg, torch::nn::ModuleList& layers, TitanFSDPManager& fsdp) 
        : fsdp_manager(&fsdp), 
          lr(cfg.learning_rate), 
          weight_decay(0.1), 
          eps(1e-8) {
        
        this->rank = torch::distributed::get_rank();
        this->world_size = torch::distributed::get_world_size();

        // ------------------------------------------------
        // 1. PARAMETER SHARDING (ZeRO-3 Core)
        // ------------------------------------------------
        // প্রতিটি GPU শুধুমাত্র তার নিজস্ব র‍্যাঙ্ক অনুযায়ী মডেলের অংশ রাখবে।
        for (auto& layer : layers) {
            for (auto& p : layer->parameters()) {
                if (!p.defined()) continue;

                // প্য়ারামিটারগুলোকে ফ্ল্যাট করে ওয়ার্ল্ড সাইজ দিয়ে ভাগ করা
                auto flattened = p.view(-1);
                int64_t total_elements = flattened.size(0);
                int64_t shard_size = (total_elements + world_size - 1) / world_size;
                
                int64_t start = rank * shard_size;
                int64_t end = std::min(start + shard_size, total_elements);

                if (start < total_elements) {
                    auto shard = flattened.slice(0, start, end).detach().clone();
                    shard.set_requires_grad(true);
                    local_param_shards.push_back(shard);
                }
            }
        }

        // 

        // ------------------------------------------------
        // 2. OPTIMIZER STATE SHARDING
        // ------------------------------------------------
        // AdamW optimizer শুধুমাত্র লোকাল শার্ডগুলোর জন্য স্টেট (m, v) তৈরি করবে।
        auto options = torch::optim::AdamWOptions(lr)
                            .betas({0.9, 0.95})
                            .eps(eps)
                            .weight_decay(weight_decay);
        
        optimizer = std::make_unique<torch::optim::AdamW>(local_param_shards, options);

        if (rank == 0) {
            std::cout << "[TitanCore] ZeRO-3 Optimizer Active." << std::endl;
            std::cout << " > Sharding: Stage 3 (Params + Grads + States)" << std::endl;
            std::cout << " > Local Shard Count: " << local_param_shards.size() << std::endl;
        }
    }

    // ------------------------------------------------
    // 3. EXECUTION STEP
    // ------------------------------------------------
    void step(torch::nn::ModuleList& layers) {
        
        // A. Synchronize Gradients (Reduce-Scatter)
        // সব GPU থেকে গ্রাডিয়েন্ট সংগ্রহ করে গড় করা এবং শার্ডে ভাগ করা।
        fsdp_manager->reduce_scatter_gradients(layers);

        // B. Optimizer Step
        // শুধুমাত্র লোকাল প্যারামিটার শার্ড আপডেট করা হয়।
        optimizer->step();

        // C. All-Gather Parameters
        // পরবর্তী ফরওয়ার্ড পাসের জন্য আপডেট হওয়া লোকাল শার্ডগুলো সব GPU-তে ব্রডকাস্ট করা।
        fsdp_manager->all_gather_parameters(layers, local_param_shards);

        // D. Clear Gradients to save VRAM
        optimizer->zero_grad();
    }

    // Adjust Learning Rate for Schedulers
    void set_lr(float new_lr) {
        for (auto& group : optimizer->param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(new_lr);
        }
    }

    // Memory Statistics
    void print_memory_stats() {
        if (rank == 0) {
            size_t total_bytes = 0;
            for (const auto& t : local_param_shards) {
                total_bytes += t.nbytes();
            }
            // AdamW stores 2 states per param (m, v) plus the param itself
            std::cout << "[ZeRO-3] Local VRAM usage (States): " 
                      << (total_bytes * 3) / (1024 * 1024) << " MB" << std::endl;
        }
    }
};
