#include <torch/torch.h>
#include <torch/distributed.h>
#include <iostream>
#include <vector>
#include <string>
#include "config.h"
#include "../../core/model/gpt.cpp"
#include "../../core/dataloader/dataset.cpp"
#include "../../distributed/nccl.cpp"
#include "../../distributed/fsdp.cpp"
#include "../../core/optimizer/zero.cpp"
#include "../../core/model/kv_cache_paged.cpp"
#include "../../safety/moderation.cpp"
#include "../../logging/audit.cpp"

/*
=====================================================
  TITANCORE: MASTER ORCHESTRATOR & TRAINING LOOP
=====================================================
  Description:
  - 3D Parallelism: ZeRO-3 + Pipeline + Tensor
  - Safety Engine: Multi-vector Moderation
  - Audit Trail: Secure Logging
=====================================================
*/

int main(int argc, char** argv) {
    // ------------------------------------------------
    // 1. Initialize Distributed Environment (NCCL)
    // ------------------------------------------------
    torch::distributed::init_process_group(torch::distributed::Backend::NCCL);
    int rank = torch::distributed::get_rank();
    int world_size = torch::distributed::get_world_size();
    
    // Set device for current rank
    c10::cuda::set_device(rank % torch::cuda::device_count());

    // ------------------------------------------------
    // 2. Load Configuration & Safety Setup
    // ------------------------------------------------
    TitanConfig cfg;
    load_config("gpt4o.yaml", cfg);

    // Initialize Safety & Audit Engine
    TitanModeration safety;
    TitanAuditLogger audit("logs/security_audit.csv");

    // ------------------------------------------------
    // 3. Initialize Advanced Communicator & FSDP
    // ------------------------------------------------
    init_nccl(rank, world_size);
    auto comm = get_nccl();
    
    // FSDP Manager for Parameter Sharding (ZeRO-3)
    TitanFSDPManager fsdp_manager(cfg, comm);

    // ------------------------------------------------
    // 4. Initialize Components (KV Cache, Model, Optimizer)
    // ------------------------------------------------
    // Paged KV Cache Manager for VRAM optimization
    PagedCacheConfig cache_cfg;
    cache_cfg.max_num_blocks = cfg.max_blocks;
    cache_cfg.n_head = cfg.n_head;
    cache_cfg.head_dim = cfg.n_embd / cfg.n_head;
    KVCacheManagerPaged kv_cache(cache_cfg);

    // Build Model (Sharded across pipeline stages)
    TitanGPT model(cfg);

    // ZeRO-3 Optimizer (Shards states, gradients, and parameters)
    TitanZeRO3 optimizer(cfg, model.blocks, fsdp_manager);
    
    // ------------------------------------------------
    // 5. Load Dataset (Distributed Sharding)
    // ------------------------------------------------
    TitanDataset dataset(cfg.dataset_path);

    

    // ------------------------------------------------
    // 6. Training Loop (500T Token Journey)                
    // ------------------------------------------------
    if (rank == 0) std::cout << "[TitanCore] Training Cycle Started..." << std::endl;

    for (int step = 0; step < cfg.max_steps; ++step) {
        // A. Load Data (Sharded)
        auto batch = dataset.get_batch(cfg.batch_size, cfg.seq_len, rank);
        auto input = batch.first; 
        auto targets = batch.second;

        // B. Safety Check: Moderation before forward pass
        if (!safety.is_safe(input)) {
            audit.log(step, rank, "UNSAFE_PROMPT_DETECTED");
            continue; // Skip dangerous batch
        }

        // C. -- Forward Pass (Pipeline Parallel) --
        torch::Tensor logits = model.forward(input, &kv_cache, step);

        // D. -- Loss Calculation --
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, cfg.vocab_size}), 
            targets.view({-1})
        );
        
        // E. -- Backward Pass & Optimizer (ZeRO-3) --
        loss.backward();
        optimizer.step(model.blocks); // Includes gradient synchronization

        // F. Logging
        if (rank == 0 && step % 100 == 0) {
            std::cout << "[Step " << step << "] Loss: " << loss.item<float>() << std::endl;
            audit.log(step, rank, std::to_string(loss.item<float>()));
        }
    }

    // ------------------------------------------------
    // 7. Cleanup
    // ------------------------------------------------
    if (rank == 0) std::cout << "[TitanCore] Task Completed." << std::endl;
    
    return 0;
}
