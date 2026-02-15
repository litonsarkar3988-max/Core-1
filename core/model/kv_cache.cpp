#pragma once
#include <torch/torch.h>
#include <vector>
#include <unordered_map>
#include <iostream>

/*
=====================================================
  TITANCORE: ULTRA PAGED KV CACHE (Multi-Layer)
=====================================================
  Optimized for 1T+ models and 120+ layers.
  Implements Logical -> Physical block mapping.
=====================================================
*/

const int64_t BLOCK_SIZE = 16;

struct PagedCacheConfig {
    int64_t max_num_blocks;  
    int64_t n_layer;         
    int64_t n_head;
    int64_t head_dim;
    torch::Device device = torch::kCUDA;
    torch::ScalarType dtype = torch::kHalf; 
};

class KVCacheManagerPaged {
private:
    // Physical Memory Pool: [Layers, Blocks, Block_Size, Heads, Dim]
    torch::Tensor K_pool;
    torch::Tensor V_pool;

    std::vector<int64_t> free_blocks;    
    // session_id -> physical block indices
    std::unordered_map<int64_t, std::vector<int64_t>> block_tables; 
    std::unordered_map<int64_t, int64_t> context_lens;              
    PagedCacheConfig config;

public:
    KVCacheManagerPaged(PagedCacheConfig cfg) : config(cfg) {
        auto options = torch::TensorOptions().dtype(cfg.dtype).device(cfg.device);

        // Huge memory allocation for all layers to avoid dynamic allocation during inference
        K_pool = torch::empty({cfg.n_layer, cfg.max_num_blocks, BLOCK_SIZE, cfg.n_head, cfg.head_dim}, options);
        V_pool = torch::empty({cfg.n_layer, cfg.max_num_blocks, BLOCK_SIZE, cfg.n_head, cfg.head_dim}, options);

        // Initialize free block list
        free_blocks.reserve(cfg.max_num_blocks);
        for (int64_t i = cfg.max_num_blocks - 1; i >= 0; --i)
            free_blocks.push_back(i);

        std::cout << "[TitanCore] KV Cache Initialized: " << cfg.n_layer << " Layers | "
                  << cfg.max_num_blocks << " Blocks per Layer." << std::endl;
    }

    

    // Allocate a new block from the free list
    int64_t allocate_block() {
        if (free_blocks.empty())
            throw std::runtime_error("KV Cache OOM: No free blocks left!");
        int64_t idx = free_blocks.back();
        free_blocks.pop_back();
        return idx;
    }

    // Append single token (zero-copy)
    void append(int64_t session_id, int64_t layer_idx, torch::Tensor k, torch::Tensor v) {
        if (block_tables.find(session_id) == block_tables.end()) {
            block_tables[session_id] = {};
            context_lens[session_id] = 0;
        }

        int64_t cur_len = context_lens[session_id];
        int64_t logical_block_idx = cur_len / BLOCK_SIZE;
        int64_t offset_in_block = cur_len % BLOCK_SIZE;

        // Allocate new block if current block is full
        if (offset_in_block == 0) {
            int64_t new_block = allocate_block();
            block_tables[session_id].push_back(new_block);
        }

        int64_t physical_block_idx = block_tables[session_id][logical_block_idx];

        using namespace torch::indexing;
        // Assume k, v shape: [1, 1, n_head, head_dim]
        auto k_sq = k.squeeze(0).squeeze(0);
        auto v_sq = v.squeeze(0).squeeze(0);

        // Write to physical memory pool
        K_pool.index_put_({layer_idx, physical_block_idx, offset_in_block, Slice(), Slice()}, k_sq);
        V_pool.index_put_({layer_idx, physical_block_idx, offset_in_block, Slice(), Slice()}, v_sq);

        // Increment context length only after last layer has appended
        if (layer_idx == config.n_layer - 1) {
            context_lens[session_id]++;
        }
    }

    // Fetches the block table for a session (essential for CUDA kernels)
    torch::Tensor get_block_table(int64_t session_id) {
        auto& blocks = block_tables[session_id];
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(config.device);
        return torch::from_blob((void*)blocks.data(), {(int64_t)blocks.size()}, options).clone();
    }

    // Free session memory and return blocks to free list
    void free_session(int64_t session_id) {
        if (block_tables.count(session_id)) {
            for (auto idx : block_tables[session_id]) free_blocks.push_back(idx);
            block_tables.erase(session_id);
            context_lens.erase(session_id);
        }
    }

    int64_t seq_len(int64_t session_id) { 
        return context_lens.count(session_id) ? context_lens[session_id] : 0; 
    }
};
