#pragma once
#include <sentencepiece_processor.h>
#include <torch/torch.h>
#include <string>
#include <vector>

/*
=====================================================
 TITANCORE SENTENCEPIECE TOKENIZER
 Production wrapper
=====================================================
*/

class TitanTokenizer {

    sentencepiece::SentencePieceProcessor sp;

public:

    TitanTokenizer(const std::string& model_path) {
        auto status = sp.Load(model_path);
        if (!status.ok()) {
            throw std::runtime_error("Failed loading tokenizer model");
        }
    }

    // ======================================
    // Encode text → token ids
    // ======================================

    std::vector<int> encode(const std::string& text) {

        std::vector<int> ids;
        sp.Encode(text, &ids);

        return ids;
    }

    // ======================================
    // Decode tokens → text
    // ======================================

    std::string decode(const std::vector<int>& tokens) {
        std::string out;
        sp.Decode(tokens, &out);
        return out;
    }

    // ======================================
    // GPU Tensor interface
    // ======================================

    torch::Tensor encode_tensor(
        const std::string& text,
        torch::Device device
    ) {

        auto ids = encode(text);

        auto tensor = torch::tensor(ids, torch::kInt64)
                        .unsqueeze(0)
                        .to(device);

        return tensor;
    }

    // ======================================
    // Special tokens
    // ======================================

    int bos_id() { return sp.bos_id(); }
    int eos_id() { return sp.eos_id(); }
    int pad_id() { return sp.pad_id(); }

};
