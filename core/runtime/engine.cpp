#include <torch/torch.h>
#include <memory>
#include "../core/model/gpt.cpp"
#include "../vision/vit.cpp"
#include "../audio/whisper.cpp"
#include "../transformer/clip.cpp"
#include "../core/sampler.cpp"
#include "../core/kv_cache.cpp"
#include "../core/fusion.cpp"

/*
====================================================
 TITANCORE RUNTIME ENGINE
 Multimodal GPT Inference Core
====================================================
*/

struct TitanEngine {

    std::shared_ptr<TitancoreGPT> gpt;
    std::shared_ptr<TitanViT> vit;
    std::shared_ptr<TitanWhisper> whisper;

    Sampler sampler;
    KVCacheManager kv;

    torch::Device device;

    TitanEngine(const TitanConfig& cfg) : sampler(cfg) {

        device = torch::cuda::is_available() ?
            torch::Device(torch::kCUDA) :
            torch::Device(torch::kCPU);

        gpt = std::make_shared<TitancoreGPT>(cfg);
        vit = std::make_shared<TitanViT>();
        whisper = std::make_shared<TitanWhisper>(cfg);

        gpt->to(device);
        vit->to(device);
        whisper->to(device);
    }

    /* ------------ TEXT ONLY ------------ */

    torch::Tensor infer_text(torch::Tensor tokens) {

        tokens = tokens.to(device);

        return gpt->generate(tokens, 128);
    }

    /* ------------ IMAGE + TEXT ------------ */

    torch::Tensor infer_vision(torch::Tensor img, torch::Tensor tokens) {

        auto vis = vit->forward(img.to(device));

        auto fused = fuse_modalities(vis, tokens.to(device));

        return gpt->generate(fused, 128);
    }

    /* ------------ AUDIO + TEXT ------------ */

    torch::Tensor infer_audio(torch::Tensor audio, torch::Tensor tokens) {

        auto aud = whisper->forward(audio.to(device));

        auto fused = fuse_modalities(aud, tokens.to(device));

        return gpt->generate(fused, 128);
    }

    /* ------------ STREAMING ------------ */

    void stream(torch::Tensor tokens,
                std::function<void(int)> cb) {

        tokens = tokens.to(device);

        for(int i=0;i<256;i++) {

            auto logits = gpt->forward(tokens, &kv, 0);

            auto next = sampler.sample(logits);

            cb(next.item<int>());

            tokens = torch::cat({tokens,next.unsqueeze(0)},1);
        }
    }
};
