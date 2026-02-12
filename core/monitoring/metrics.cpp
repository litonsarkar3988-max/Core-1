#pragma once
#include <atomic>
#include <chrono>

/*
====================================
 TITANCORE CORE METRICS ENGINE
====================================
*/

struct TitanCoreMetrics {

    std::atomic<long> active_sessions{0};
    std::atomic<long> tokens_generated{0};
    std::atomic<long> requests{0};
    std::atomic<long> errors{0};

    std::atomic<long> gpu_vram_mb{0};
    std::atomic<long> kv_cache_mb{0};
    std::atomic<long> queue_depth{0};

    std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();

    void on_request() {
        requests++;
    }

    void on_session_start() {
        active_sessions++;
    }

    void on_session_end() {
        active_sessions--;
    }

    void on_token() {
        tokens_generated++;
    }

    void on_error() {
        errors++;
    }

    double tokens_per_second() {

        auto now = std::chrono::steady_clock::now();

        double secs =
            std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

        if (secs == 0) return 0;

        return tokens_generated.load() / secs;
    }
};
