#include <string>
#include <vector>
#include <iostream>
#include <regex>

/*
=========================================
 TITANCORE JAILBREAK DETECTOR
 Detects:
  - Prompt injection
  - System override
  - Role hijack
  - DAN prompts
=========================================
*/

struct JailbreakResult {
    bool detected;
    float confidence;
    std::string rule;
};

class TitanJailbreak {

private:

    // Common jailbreak / injection patterns
    std::vector<std::regex> rules = {
        std::regex("ignore previous", std::regex::icase),
        std::regex("system prompt", std::regex::icase),
        std::regex("developer mode", std::regex::icase),
        std::regex("act as", std::regex::icase),
        std::regex("you are now", std::regex::icase),
        std::regex("bypass", std::regex::icase),
        std::regex("dan", std::regex::icase),
        std::regex("jailbreak", std::regex::icase)
    };

public:

    JailbreakResult scan(const std::string& input) {

        for (auto& rule : rules) {
            if (std::regex_search(input, rule)) {
                return {
                    true,
                    0.85f,
                    rule.mark_count() ? "regex" : "keyword"
                };
            }
        }

        return {false, 0.01f, "clean"};
    }

    bool is_safe(const std::string& text) {

        auto res = scan(text);

        if (res.detected) {
            std::cout << "[Jailbreak] Prompt attack detected\n";
            return false;
        }

        return true;
    }
};
