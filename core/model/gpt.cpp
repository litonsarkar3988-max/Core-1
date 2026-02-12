#include <torch/torch.h>
#include <vector>
#include <memory>

#include "config.h"
#include "block.cpp"
#include "kv_cache.cpp"

using Tensor = torch::Tensor;

/*
====================================================
 TITANCORE GPT DECODER (Inference Engine)
====================================================
*/

struct TitancoreGPT : torch::nn::Module {

    torch::nn::Embedding wte{nullptr};
    torch::nn::Embedding wpe{nullptr};

    std::vector<std::shared_ptr<TransformerBlock>> blocks;

    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};

    int64_t block_size;

    TitancoreGPT(const TitanConfig& cfg) {

        block_size = cfg.block_size;

        // Token embedding
        wte = register_module("wte",
            torch::nn::Embedding(cfg.vocab_size, cfg.n_embd));

        // Position embedding
        wpe = register_module("wpe",
            torch::nn::Embedding(cfg.block_size, cfg.n_embd));

        // Transformer blocks
        for(int i=0;i<cfg.n_layer;i++) {

            auto block = std::make_shared<TransformerBlock>(cfg);

            blocks.push_back(block);

            register_module("block_"+std::to_string(i), block);
        }

        // Final norm
        ln_f = register_module("ln_f",
            torch::nn::LayerNorm(cfg.n_embd));

        // Language head
        lm_head = register_module("lm_head",
            torch::nn::Linear(cfg.n_embd, cfg.vocab_size));
    }

    /*
    --------------------------------------
     Forward pass (KV cached inference)
    --------------------------------------
    */

    Tensor forward(Tensor tokens,
                   KVCacheManager* cache,
                   int64_t session_id) {

        auto B = tokens.size(0);
        auto T = tokens.size(1);

        auto device = tokens.device();

        // Positional ids
        auto pos = torch::arange(0, T,
                     torch::TensorOptions().dtype(torch::kLong))
                     .to(device);

        // Embedding
        Tensor x = wte(tokens) + wpe(pos);

        // Decoder blocks
        for(auto& block : blocks) {
            x = block->forward(x, cache, session_id);
        }

        // Final norm
        x = ln_f(x);

        // Logits
        return lm_head(x);
    }
};
