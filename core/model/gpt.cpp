#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include "config.h"
#include "embedding.cpp"
#include "block.cpp"
#include "kv_cache_paged.cpp"
#include "../../distributed/nccl.cpp"

/*
====================================================
  TITANCORE: GPT-4o ULTRA (TRILLION SCALE)
====================================================
  Integrated 3D Parallelism:
  - Tensor Parallelism (Sharded Embeddings & Attention)
  - Pipeline Parallelism (Layer Partitioning)
  - Data Parallelism (FSDP / Sharded States)
====================================================
*/

struct TitanGPT : torch::nn::Module {
    std::shared_ptr<TitanParallelEmbedding> tok_emb{nullptr};
    torch::nn::ModuleList blocks;
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear head{nullptr};

    int64_t n_layer;
    int rank, pipeline_rank, pipeline_world_size;

    TitanGPT(const TitanConfig& cfg) {
        n_layer = cfg.n_layer;
        
        auto comm = get_nccl();
        rank = comm->rank;
        int total_world_size = torch::distributed::get_world_size();
        
        pipeline_world_size = cfg.pipeline_parallel_size; 
        int gpus_per_stage = total_world_size / pipeline_world_size;
        pipeline_rank = rank / gpus_per_stage;

        // Stage 0: Parallel Embedding
        if (pipeline_rank == 0) {
            tok_emb = register_module("tok_emb", std::make_shared<TitanParallelEmbedding>(cfg));
        }

        // Layer Partitioning for Pipeline Parallelism
        int layers_per_stage = n_layer / pipeline_world_size;
        int start_layer = pipeline_rank * layers_per_stage;
        int end_layer = start_layer + layers_per_stage;

        

        for (int i = start_layer; i < end_layer; ++i) {
            auto block = std::make_shared<TransformerBlockMoE>(cfg);
            blocks->push_back(register_module("block_" + std::to_string(i), block));
        }

        // Final Stage: Norm and Language Model Head
        if (pipeline_rank == pipeline_world_size - 1) {
            ln_f = register_module("ln_f", torch::nn::LayerNorm(cfg.n_embd, 1e-5));
            head = register_module("head", torch::nn::Linear(cfg.n_embd, cfg.vocab_size, false));
        }

        this->to(torch::kCUDA);
        this->to(torch::kFloat16);
    }

    torch::Tensor forward(torch::Tensor tokens, 
                          KVCacheManagerPaged* cache, 
                          int64_t session_id) {
        
        torch::Tensor x;
        auto comm = get_nccl();
        int gpus_per_stage = torch::distributed::get_world_size() / pipeline_world_size;

        // 1. Input Handling
        if (pipeline_rank == 0) {
            x = tok_emb->forward(tokens);
        } else {
            // Receive hidden states from previous pipeline stage
            x = torch::empty({tokens.size(0), tokens.size(1), 16384}, 
                             torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
            comm->recv(x, rank - gpus_per_stage);
        }

        // 2. Transformer Blocks Processing
        for (int i = 0; i < blocks->size(); ++i) {
            int64_t global_layer_idx = (pipeline_rank * (n_layer / pipeline_world_size)) + i;
            x = blocks[i]->as<TransformerBlockMoE>()->forward(x, cache, session_id, global_layer_idx);
        }

        // 3. Output Handling
        if (pipeline_rank == pipeline_world_size - 1) {
            x = ln_f(x);
            return head(x);
        } else {
            // Send hidden states to next pipeline stage
            comm->send(x, rank + gpus_per_stage);
            return x; 
        }
    }
};
