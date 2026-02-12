#include <torch/torch.h>
#include <iostream>
#include <unordered_map>
#include <string>

/*
=========================================
 TITANCORE MODEL LOADER
 Loads:
  - GPT weights
  - Vision weights
  - Whisper weights
  - Quantized shards
=========================================
*/

struct TitanLoader {

    torch::Device device;

    TitanLoader(torch::Device dev) : device(dev) {}

    // Load full model
    template<typename T>
    std::shared_ptr<T> load_model(const std::string& path) {

        auto model = std::make_shared<T>();

        try {
            torch::load(model, path);
        }
        catch (...) {
            std::cerr << "[Loader] Failed loading: " << path << std::endl;
            exit(1);
        }

        model->to(device);
        model->eval();

        std::cout << "[Loader] Model loaded: " << path << std::endl;

        return model;
    }

    // Load shard (FSDP / tensor parallel)
    torch::Tensor load_shard(const std::string& shard_path) {

        torch::Tensor shard;

        torch::load(shard, shard_path);

        std::cout << "[Loader] Shard loaded: " << shard_path << std::endl;

        return shard.to(device);
    }

    // Quantized weights
    torch::Tensor load_quant(const std::string& path) {

        torch::Tensor q;

        torch::load(q, path);

        std::cout << "[Loader] Quantized tensor loaded" << std::endl;

        return q.to(device);
    }
};
