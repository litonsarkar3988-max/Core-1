#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <mutex>
#include "../safety/moderation.cpp"

/*
=====================================================
  TITANCORE: SECURITY AUDIT & LOGGING ENGINE
=====================================================
  - Asynchronous Logging
  - Structured Audit Trail for Safety Violations
  - Data Retention for Future Fine-tuning
=====================================================
*/

class TitanAuditLogger {
private:
    std::string log_file;
    std::mutex log_mutex;

    // Helper to get current timestamp
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        return ss.str();
    }

public:
    TitanAuditLogger(const std::string& filename) : log_file(filename) {
        std::ofstream file(log_file, std::ios::app);
        file << "Timestamp,SessionID,ViolationType,Confidence,RawText\n";
        std::cout << "[TitanCore] Audit Logger Initialized: " << log_file << std::endl;
    }

    // 

    void log_violation(int64_t session_id, const ModerationResult& result, const std::string& text) {
        std::lock_guard<std::mutex> lock(log_mutex);
        
        std::ofstream file(log_file, std::ios::app);
        
        // Convert enum to string
        std::string violation_type;
        switch(result.label) {
            case SafetyLabel::VIOLENCE: violation_type = "VIOLENCE"; break;
            case SafetyLabel::NSFW: violation_type = "NSFW"; break;
            case SafetyLabel::HATE: violation_type = "HATE"; break;
            case SafetyLabel::MALWARE: violation_type = "MALWARE"; break;
            case SafetyLabel::PROMPT_INJECTION: violation_type = "INJECTION"; break;
            case SafetyLabel::JAILBREAK: violation_type = "JAILBREAK"; break;
            default: violation_type = "UNKNOWN";
        }

        // Sanitize text (remove commas/newlines for CSV format)
        std::string sanitized_text = text;
        std::replace(sanitized_text.begin(), sanitized_text.end(), ',', ' ');
        std::replace(sanitized_text.begin(), sanitized_text.end(), '\n', ' ');

        file << get_timestamp() << ","
             << session_id << ","
             << violation_type << ","
             << result.confidence << ","
             << sanitized_text << "\n";
             
        std::cout << "[Audit] Violation Logged: " << violation_type << " | Session: " << session_id << std::endl;
    }
};
