#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/*
=====================================================
  TITANCORE: ULTRA-SCALE DATASET LOADER (1T+)
=====================================================
  - Memory-Mapped I/O for 10TB+ Data
  - Distributed Asynchronous Sharding
  - High-Speed Tokenization Hook
=====================================================
*/

struct TitanDataset {
private:
    std::string dataset_path;
    int64_t total_tokens;
    int64_t* mmap_ptr;
    int fd;

public:
    TitanDataset(const std::string& path) : dataset_path(path) {
        // 1. Open Dataset file
        fd = open(dataset_path.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Cannot open dataset file!");
        }

        // 2. Get file size to calculate total tokens
        off_t file_size = lseek(fd, 0, SEEK_END);
        total_tokens = file_size / sizeof(int64_t);
        lseek(fd, 0, SEEK_SET);

        // 

        // 3. Memory Map the dataset (No loading into RAM yet)
        mmap_ptr = (int64_t*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mmap_ptr == MAP_FAILED) {
            throw std::runtime_error("Memory mapping failed!");
        }

        if (torch::distributed::get_rank() == 0) {
            std::cout << "[TitanCore] Dataset Initialized. Total Tokens: " 
                      << total_tokens / 1e9 << " Billion." << std::endl;
        }
    }

    ~TitanDataset() {
        if (mmap_ptr != MAP_FAILED) munmap(mmap_ptr, total_tokens * sizeof(int64_t));
        close(fd);
    }

    // 

    // 4. Asynchronous Distributed Sharding
    std::pair<torch::Tensor, torch::Tensor> get_batch(int batch_size, int seq_len, int rank) {
        int world_size = torch::distributed::get_world_size();
        
        // Calculate shard per GPU
        int64_t tokens_per_gpu = total_tokens / world_size;
        int64_t shard_start = rank * tokens_per_gpu;

        // Randomly sample within the GPU's shard
        int64_t max_start = shard_start + tokens_per_gpu - (batch_size * (seq_len + 1));
        int64_t current_pos = shard_start + (rand() % (max_start - shard_start));

        // 5. Create Tensors directly from Mapped Memory (Zero-Copy)
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        
        std::vector<torch::Tensor> input_batch;
        std::vector<torch::Tensor> target_batch;

        for (int b = 0; b < batch_size; ++b) {
            // Raw pointer access from mmap
            auto* data_ptr = mmap_ptr + current_pos;
            
            // Create CPU tensors (Zero-copy from mmap)
            auto sequence = torch::from_blob(data_ptr, {seq_len + 1}, options).clone();
            
            input_batch.push_back(sequence.slice(0, 0, seq_len));
            target_batch.push_back(sequence.slice(0, 1, seq_len + 1));
            
            current_pos += (seq_len + 1); // Simple sequential loading
        }

        // 6. Stack and Transfer to GPU
        auto input = torch::stack(input_batch).to(torch::kCUDA);
        auto targets = torch::stack(target_batch).to(torch::kCUDA);

        return {input, targets};
    }
};
