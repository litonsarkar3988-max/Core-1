#include <torch/torch.h>
#include "../vision/vit.cpp"
#include "../core/model/embedding.cpp"

/*
====================================================
 TITANCORE CLIP
 Vision ↔ Text Alignment Module
====================================================
*/

struct TitanCLIP : torch::nn::Module {

    // Vision
    TitanViT vision{nullptr};

    // Text
    TitanEmbedding text_embed{nullptr};

    // Projection heads
    torch::nn::Linear vision_proj{nullptr};
    torch::nn::Linear text_proj{nullptr};

    int dim;

    TitanCLIP(const TitanConfig& cfg)
        : vision(224,16,cfg.n_embd,12,cfg.n_head),
          text_embed(cfg),
          dim(cfg.n_embd)
    {
        register_module("vision", vision);
        register_module("text_embed", text_embed);

        vision_proj = register_module("vision_proj",
            torch::nn::Linear(cfg.n_embd,cfg.n_embd));

        text_proj = register_module("text_proj",
            torch::nn::Linear(cfg.n_embd,cfg.n_embd));
    }

    // Encode image → latent
    torch::Tensor encode_image(torch::Tensor img) {

        auto x = vision.forward(img);     // [B,T,C]
        x = x.mean(1);                   // pool
        return torch::nn::functional::normalize(
            vision_proj(x),
            torch::nn::functional::NormalizeFuncOptions().dim(1)
        );
    }

    // Encode text → latent
    torch::Tensor encode_text(torch::Tensor tokens) {

        auto x = text_embed.forward(tokens); // [B,T,C]
        x = x.mean(1);

        return torch::nn::functional::normalize(
            text_proj(x),
            torch::nn::functional::NormalizeFuncOptions().dim(1)
        );
    }

    // CLIP loss (contrastive)
    torch::Tensor forward(torch::Tensor img, torch::Tensor text) {

        auto I = encode_image(img);
        auto T = encode_text(text);

        auto logits = torch::matmul(I, T.t()) * 100.0;

        auto labels = torch::arange(0, logits.size(0), torch::kLong).to(img.device());

        return torch::nn::functional::cross_entropy(logits, labels);
    }
};
