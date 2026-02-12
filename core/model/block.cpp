#include <torch/torch.h>
#include <vector>

#include "config.h"
#include "kv_cache.cpp"
#include "mlp.cpp"

// CUDA FlashAttention binding
torch::Tensor flash_attention(torch::Tensor Q,
                              torch::Tensor K,
                              torch::Tensor V);

/*
====================================================
 TITANCORE TRANSFORMER BLOCK
====================================================
*/

struct TransformerBlock : torch::nn::Module {

    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::LayerNorm ln2{nullptr};

    torch::nn::Linear qkv{nullptr};
    torch::nn::Linear proj{nullptr};

    MLP mlp;

    int64_t n_head;
    int64_t head_dim;

    TransformerBlock(const TitanConfig& cfg)
    : mlp(cfg) {

        n_head = cfg.n_head;
        head_dim = cfg.n_embd / cfg.n_head;

        ln1 = register_module("ln1",
            torch::nn::LayerNorm(cfg.n_embd));

        ln2 = register_module("ln2",
            torch::nn::LayerNorm(cfg.n_embd));

        qkv = register_module("qkv",
            torch::nn::Linear(cfg.n_embd, 3 * cfg.n_embd));

        proj = register_module("proj",
            torch::nn::Linear(cfg.n_embd, cfg.n_embd));

        register_module("mlp", mlp);
    }

    torch::Tensor forward(torch::Tensor x,
                          KVCacheManager* cache,
                          int64_t session_id) {

        auto B = x.size(0);
        auto T = x.size(1);
        auto C = x.size(2);

        // ===============================
        // LayerNorm + QKV projection
        // ===============================

        auto h = ln1(x);
        auto qkv_out = qkv(h);

        auto Q = qkv_out.slice(2, 0, C);
        auto K = qkv_out.slice(2, C, 2*C);
        auto V = qkv_out.slice(2, 2*C, 3*C);

        // ===============================
        // KV Cache append
        // ===============================

        if(cache) {
            cache->append(session_id, K, V);
            auto cached = cache->get(session_id);
            K = cached.first;
            V = cached.second;
        }

        // ===============================
        // Flash Attention
        // ===============================

        auto attn = flash_attention(Q, K, V);

        attn = proj(attn);

        // Residual
        x = x + attn;

        // ===============================
        // MLP
        // ===============================

        auto y = ln2(x);
        y = mlp.forward(y);

        return x + y;
    }
};
