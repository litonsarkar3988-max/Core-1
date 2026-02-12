#include <torch/torch.h>
#include "config.h"

/*
====================================================
 TITANCORE EMBEDDING LAYER
 Token + Position
====================================================
*/

struct TitanEmbedding : torch::nn::Module {

    torch::nn::Embedding token_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};

    int64_t block_size;

    TitanEmbedding(const TitanConfig& cfg) {

        block_size = cfg.block_size;

        // Token embeddings
        token_emb = register_module(
            "token_embedding",
            torch::nn::Embedding(cfg.vocab_size, cfg.n_embd)
        );

        // Absolute positional embeddings
        pos_emb = register_module(
            "position_embedding",
            torch::nn::Embedding(cfg.block_size, cfg.n_embd)
        );
    }

    torch::Tensor forward(torch::Tensor tokens) {

        auto B = tokens.size(0);
        auto T = tokens.size(1);

        TORCH_CHECK(T <= block_size, "Sequence too long");

        auto device = tokens.device();

        // Position ids [0...T]
        auto pos = torch::arange(
            0, T,
            torch::TensorOptions().dtype(torch::kLong).device(device)
        );

        // Broadcast to batch
        pos = pos.unsqueeze(0).expand({B, T});

        // Embeddings
        auto tok = token_emb(tokens);
        auto posi = pos_emb(pos);

        return tok + posi;
    }
};
