#include <torch/torch.h>
#include <torch/distributed.h>
#include <iostream>
#include <vector>
#include "config.h"
#include "core/model/gpt.cpp"
#include "core/dataloader/dataset.cpp"
#include "distributed/nccl.cpp"
#include "distributed/fsdp.cpp"
#include "core/optimizer/zero.cpp"
#include "core/model/kv_cache_paged.cpp"

/*
====================================================
  TITANCORE: MAIN ENGINE & TRAINING LOOP
====================================================
  Implements 3D Parallelism for 1T+ Models.
====================================================
*/

int main(int argc, char** argv) {
    // 1. Initialize Distributed Environment
    torch::distributed::init_process_group(torch::distributed::Backend::NCCL);
    int rank = torch::distributed::get_rank();
    int world_size = torch::distributed::get_world_size();
    
    // Set device
    c10::cuda::set_device(rank % torch::cuda::device_count());

    // 2. Load Configuration
    TitanConfig cfg;
    load_config("gpt4o.yaml", cfg);

    // 3. Initialize Advanced Communicator
    init_nccl(rank, world_size);
    auto comm = get_nccl();

    // 4. Initialize Pipeline & FSDP Manager
    TitanFSDPManager fsdp_manager(cfg);

    // 5. Initialize Paged KV Cache Manager
    PagedCacheConfig cache_cfg;
    cache_cfg.max_num_blocks = cfg.max_blocks;
    cache_cfg.n_head = cfg.n_head;
    cache_cfg.head_dim = cfg.n_embd / cfg.n_head;
    KVCacheManagerPaged kv_cache(cache_cfg);

    // 6. Build Model
    TitanGPT model(cfg);

    // 7. Initialize ZeRO Optimizer
    ZeROOptimizer optimizer(cfg, model.blocks, fsdp_manager);
    
    // 8. Load Dataset
    TitanDataset dataset(cfg.dataset_path);

    // 9. Training Loop (500T Token Journey)                
    for (int step = 0; step < cfg.max_steps; ++step) {
        // Load Data (Assume data_parallel sharding here)
        auto batch = dataset.get_batch(cfg.batch_size, cfg.seq_len, rank);
        auto input = batch.first; 
        auto targets = batch.second;

        // -- Forward Pass (Pipeline Parallel) --
        torch::Tensor logits = model.forward(input, &kv_cache, step);

        // -- Loss Calculation --
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, cfg.vocab_size}), 
            targets.view({-1})
        );
        
        // -- Backward Pass & Optimizer --
        loss.backward();
        optimizer.step();

        // Logging
        if (rank == 0 && step % 100 == 0) {
            std::cout << "Step: " << step << " | Loss: " << loss.item<float>() << std::endl;
        }
    }

    return 0;
}
