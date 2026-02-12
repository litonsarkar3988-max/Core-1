#include <torch/torch.h>
#include "../vision/vit.cpp"
#include "../core/model/embedding.cpp"
#include "../distributed/fsdp.cpp"
#include "../quant/int8.cpp"
#include "../quant/int4.cpp"
#include "../quant/fp8.cpp"

/*
====================================================
 TITANCORE CLIP (GPU + Quantization + FSDP)
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

    // FSDP wrapper
    FSDP* fsdp = nullptr;

    TitanCLIP(const TitanConfig& cfg)
        : vision(224,16,cfg.n_embd,12,cfg.n_head),
          text_embed(cfg),
          dim(cfg.n_embd)
    {
        register_module("vision", vision);
        register_module("text_embed", text_embed);

        vision_proj = register_module("vision_proj",
            torch::nn::Linear(cfg.n_embd,cfg.n_embd).to(torch::kCUDA));

        text_proj = register_module("text_proj",
            torch::nn::Linear(cfg.n_embd,cfg.n_embd).to(torch::kCUDA));
    }

    // Encode image → latent
    torch::Tensor encode_image(torch::Tensor img) {

        img = img.to(torch::kCUDA);

        auto x = vision.forward(img);     // [B,T,C]
        x = x.mean(1);                     // pool

        // GPU FSDP gather
        if(fsdp) {
            auto full_weight = fsdp->gather(vision_proj->weight);
            auto qw = TitanINT8::quantize(vision_proj->weight);
            x = TitanINT8::linear(x, qw, vision_proj->bias);

            auto qw4 = TitanINT4::quantize(vision_proj->weight);
            x = TitanINT4::linear(x, qw4, vision_proj->bias);

            auto fp8_w = TitanFP8::quantize(vision_proj->weight);
            x = TitanFP8::linear(x, fp8_w, vision_proj->bias);
        } else {
            x = vision_proj->forward(x);
        }

        return torch::nn::functional::normalize(
            x,
            torch::nn::functional::NormalizeFuncOptions().dim(1)
        );
    }

    // Encode text → latent
    torch::Tensor encode_text(torch::Tensor tokens) {

        tokens = tokens.to(torch::kCUDA);

        auto x = text_embed.forward(tokens); // [B,T,C]
        x = x.mean(1);

        if(fsdp) {
            auto full_weight = fsdp->gather(text_proj->weight);
            auto qw = TitanINT8::quantize(text_proj->weight);
            x = TitanINT8::linear(x, qw, text_proj->bias);

            auto qw4 = TitanINT4::quantize(text_proj->weight);
            x = TitanINT4::linear(x, qw4, text_proj->bias);

            auto fp8_w = TitanFP8::quantize(text_proj->weight);
            x = TitanFP8::linear(x, fp8_w, text_proj->bias);
        } else {
            x = text_proj->forward(x);
        }

        return torch::nn::functional::normalize(
            x,
            torch::nn::functional::NormalizeFuncOptions().dim(1)
        );
    }

    // CLIP contrastive loss
    torch::Tensor forward(torch::Tensor img, torch::Tensor text) {

        auto I = encode_image(img);
        auto T = encode_text(text);

        auto logits = torch::matmul(I, T.t()) * 100.0;

        auto labels = torch::arange(0, logits.size(0), torch::kLong)
                          .to(img.device());

        return torch::nn::functional::cross_entropy(logits, labels);
    }
};
