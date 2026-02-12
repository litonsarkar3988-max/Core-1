#include <string>
#include <mutex>
#include <unordered_map>
#include <iostream>

/*
==========================================
 TITANCORE PROMETHEUS EXPORTER
 Metrics endpoint for monitoring
==========================================
*/

struct Metric {
    double value = 0.0;
};

class TitanMetrics {

private:
    std::mutex lock;

    std::unordered_map<std::string, Metric> metrics;

public:

    void inc(const std::string& name, double v = 1.0) {
        std::lock_guard<std::mutex> guard(lock);
        metrics[name].value += v;
    }

    void set(const std::string& name, double v) {
        std::lock_guard<std::mutex> guard(lock);
        metrics[name].value = v;
    }

    std::string render() {

        std::lock_guard<std::mutex> guard(lock);

        std::string out;

        for (auto& m : metrics) {
            out += m.first + " " + std::to_string(m.second.value) + "\n";
        }

        return out;
    }
};
