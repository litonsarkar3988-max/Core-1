#include <unordered_map>
#include <chrono>
#include <string>
#include <iostream>

/*
=========================================
 TITANCORE RATE LIMITER
 Protects against:
  - Request flooding
  - Token abuse
  - Bot hammering
=========================================
*/

struct RateState {
    int count = 0;
    std::chrono::steady_clock::time_point window_start;
};

class TitanRateLimiter {

private:

    // key = user_id / ip / session
    std::unordered_map<std::string, RateState> table;

    int max_requests;
    int window_seconds;

public:

    TitanRateLimiter(int max_req = 30, int window_sec = 60)
        : max_requests(max_req), window_seconds(window_sec) {}

    bool allow(const std::string& key) {

        auto now = std::chrono::steady_clock::now();

        auto& state = table[key];

        // First request
        if (state.count == 0) {
            state.window_start = now;
            state.count = 1;
            return true;
        }

        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(
                now - state.window_start
            ).count();

        // Reset window
        if (elapsed > window_seconds) {
            state.window_start = now;
            state.count = 1;
            return true;
        }

        // Check limit
        if (state.count >= max_requests) {
            std::cout << "[RateLimit] Blocked key: " << key << std::endl;
            return false;
        }

        state.count++;
        return true;
    }
};
