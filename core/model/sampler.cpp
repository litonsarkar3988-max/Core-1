#pragma once
#include <torch/torch.h>

/*
=====================================================
 TITANCORE SAMPLER
 Temperature + TopK + TopP (Nucleus)
=====================================================
*/

struct SamplingConfig {
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
};

class TitanSampler {

public:

    static torch::Tensor sample(
        torch::Tensor logits,
        SamplingConfig cfg
    ) {

        // logits: [B, vocab]

        if (cfg.temperature != 1.0f)
            logits = logits / cfg.temperature;

        // =========================
        // TOP-K
        // =========================

        if (cfg.top_k > 0) {

            auto topk = logits.topk(cfg.top_k);
            auto values = std::get<0>(topk);
            auto indices = std::get<1>(topk);

            auto probs = torch::softmax(values, -1);
            auto sample = torch::multinomial(probs, 1);

            return torch::gather(indices, 1, sample);
        }

        // =========================
        // TOP-P (NUCLEUS)
        // =========================

        if (cfg.top_p < 1.0f) {

            auto sorted = logits.sort(-1, true);
            auto sorted_logits = std::get<0>(sorted);
            auto sorted_indices = std::get<1>(sorted);

            auto probs = torch::softmax(sorted_logits, -1);
            auto cumulative = probs.cumsum(-1);

            auto mask = cumulative > cfg.top_p;
            mask.index_put_({torch::indexing::Slice(), 0}, false);

            sorted_logits.masked_fill_(mask, -INFINITY);

            auto filtered_probs = torch::softmax(sorted_logits, -1);
            auto sample = torch::multinomial(filtered_probs, 1);

            return torch::gather(sorted_indices, 1, sample);
        }

        // =========================
        // FULL SOFTMAX (Fallback)
        // =========================

        auto probs = torch::softmax(logits, -1);
        return torch::multinomial(probs, 1);
    }
};
