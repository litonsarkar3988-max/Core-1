#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_set>

/*
=========================================
 TITANCORE SAFETY MODERATION ENGINE
 Filters:
  - Violence
  - NSFW
  - Hate
  - Malware
  - Prompt Injection
=========================================
*/

enum class SafetyLabel {
    SAFE,
    VIOLENCE,
    NSFW,
    HATE,
    MALWARE,
    PROMPT_INJECTION
};

struct ModerationResult {
    SafetyLabel label;
    float confidence;
};

class TitanModeration {

private:

    std::unordered_set<std::string> banned_words = {
        "kill", "bomb", "terror", "sex", "hack", "suicide"
    };

public:

    // Lightweight lexical scan (fast path)
    ModerationResult quick_scan(const std::string& text) {

        for (auto& w : banned_words) {
            if (text.find(w) != std::string::npos) {
                return {SafetyLabel::PROMPT_INJECTION, 0.80};
            }
        }

        return {SafetyLabel::SAFE, 0.01};
    }

    // Deep moderation (ML classifier placeholder)
    ModerationResult deep_scan(torch::Tensor embedding) {

        // Real system would use trained safety classifier

        float score = embedding.mean().item<float>();

        if (score > 0.7)
            return {SafetyLabel::HATE, score};

        return {SafetyLabel::SAFE, score};
    }

    bool is_allowed(const std::string& text, torch::Tensor emb) {

        auto fast = quick_scan(text);
        if (fast.label != SafetyLabel::SAFE) {
            std::cout << "[Safety] Blocked by quick scan\n";
            return false;
        }

        auto deep = deep_scan(emb);
        if (deep.label != SafetyLabel::SAFE) {
            std::cout << "[Safety] Blocked by ML scan\n";
            return false;
        }

        return true;
    }
};
