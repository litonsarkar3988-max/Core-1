#pragma once
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "config.h"
#include "kv_cache_paged.cpp"

// External CUDA Kernel Bindings
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

/*
====================================================
  HELPER: ROTARY POSITIONAL EMBEDDINGS (RoPE)
====================================================
*/
torch::Tensor apply_rotary_emb(torch::Tensor x, int64_t dim) {
    auto B = x.size(0);
    auto n_head = x.size(1);
    auto T = x.size(2);
    auto head_dim = x.size(3);

    auto inv_freq = 1.0 / torch::pow(10000.0, torch::arange(0, head_dim, 2, x.options()) / head_dim);
    auto t = torch::arange(T, x.options());
    auto freqs = torch::einsum("i,j->ij", {t, inv_freq});

    auto freqs_cos = freqs.cos().repeat({1, 2});
    auto freqs_sin = freqs.sin().repeat({1, 2});

    return x * freqs_cos.unsqueeze(0).unsqueeze(0) +
           torch::cat({-x.slice(3, head_dim/2), x.slice(3, 0, head_dim/2)}, 3) * freqs_sin.unsqueeze(0).unsqueeze(0);
}

/*
====================================================
  COMPONENT: SwiGLU MLP (Standard for Llama/GPT-4)
====================================================
*/
struct SwiGLU_MLP : torch::nn::Module {
    torch::nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};

    SwiGLU_MLP(int64_t n_embd, int64_t hidden_dim) {
        w1 = register_module("w1", torch::nn::Linear(n_embd, hidden_dim, false));
        w3 = register_module("w3", torch::nn::Linear(n_embd, hidden_dim, false));
        w2 = register_module("w2", torch::nn::Linear(hidden_dim, n_embd, false));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto gate = torch::silu(w1(x));
        auto up = w3(x);
        return w2(gate * up);
    }
};

/*
====================================================
  CORE: TITANCORE TRANSFORMER BLOCK (MoE)
====================================================
*/
struct TransformerBlockMoE : torch::nn::Module {
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::Linear qkv{nullptr}, proj{nullptr};
    torch::nn::Linear gate{nullptr};
    std::vector<std::shared_ptr<SwiGLU_MLP>> experts;

    int64_t n_head, head_dim, n_experts, top_k;

    TransformerBlockMoE(const TitanConfig& cfg) {
        n_head = cfg.n_head;
        head_dim = cfg.n_embd / cfg.n_head;
        n_experts = cfg.n_experts;
        top_k = cfg.top_k;

        ln1 = register_module("ln1", torch::nn::LayerNorm(cfg.n_embd, 1e-5));
        ln2 = register_module("ln2", torch::nn::LayerNorm(cfg.n_embd, 1e-5));

        qkv = register_module("qkv", torch::nn::Linear(cfg.n_embd, 3*cfg.n_embd, false));
        proj = register_module("proj", torch::nn::Linear(cfg.n_embd, cfg.n_embd, false));

        gate = register_module("gate", torch::nn::Linear(cfg.n_embd, n_experts, false));

        int64_t hidden_dim = cfg.n_embd * 4;
        for(int i=0;i<n_experts;i++){
            auto e = std::make_shared<SwiGLU_MLP>(cfg.n_embd, hidden_dim);
            experts.push_back(e);
            register_module("expert_"+std::to_string(i), e);
        }
    }

    torch::Tensor forward(torch::Tensor x,
                          KVCacheManagerPaged* cache,
                          int64_t session_id,
                          int64_t layer_idx) {
        auto B = x.size(0);
        auto T = x.size(1);
        auto C = x.size(2);

        // ----------------------------
        // 1. Attention
        // ----------------------------
        auto residual = x;
        x = ln1(x);

        auto qkv_out = qkv(x);
        auto chunks = qkv_out.chunk(3, -1);
        auto Q = chunks[0].view({B, T, n_head, head_dim}).transpose(1,2);
        auto K = chunks[1].view({B, T, n_head, head_dim}).transpose(1,2);
        auto V = chunks[2].view({B, T, n_head, head_dim}).transpose(1,2);

        Q = apply_rotary_emb(Q, head_dim);
        K = apply_rotary_emb(K, head_dim);

        if(cache){
            cache->append(session_id, layer_idx, K, V);
            auto kv_pair = cache->get_full_kv(session_id, layer_idx);
            K = kv_pair.first;
            V = kv_pair.second;
        }

        x = residual + proj(flash_attention_forward(Q,K,V));

        // ----------------------------
        // 2. MoE
        // ----------------------------
        residual = x;
        x = ln2(x);

        auto router_logits = gate(x);
        auto router_probs = torch::softmax(router_logits, -1);
        auto topk = torch::topk(router_probs, top_k, -1);
        auto routing_weights = std::get<0>(topk);
        auto selected_experts = std::get<1>(topk);

        routing_weights /= routing_weights.sum(-1, true);

        auto final_output = torch::zeros_like(x);

        for(int i=0;i<n_experts;i++){
            auto mask = (selected_experts==i).any(-1);
            if(mask.any().item<bool>()){
                auto expert_out = experts[i]->forward(x);
                auto weight_mask = (selected_experts==i);
                auto weight_for_expert = (routing_weights*weight_mask).sum(-1).unsqueeze(-1);
                final_output += expert_out*weight_for_expert*mask.unsqueeze(-1);
            }
        }

        return residual + final_output;
    }
