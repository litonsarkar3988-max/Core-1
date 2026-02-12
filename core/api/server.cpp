// server.cpp
#include "crow_all.h"                 // Crow single-header web framework
#include <torch/torch.h>

#include "../runtime/engine.cpp"
#include "../safety/jailbreak.cpp"
#include "../safety/moderation.cpp"
#include "../safety/rate_limit.cpp"
#include "../retrieval/embedder.cpp"
#include "../api/auth.cpp"
#include "../monitoring/prometheus.cpp"
#include "../monitoring/metrics.cpp"

using json = crow::json::wvalue;

TitanCoreMetrics core_metrics;  // global metrics instance

int main() {

    crow::SimpleApp app;

    // ======== Services ========
    TitanRateLimiter rate;
    TitanJailbreak jailbreak;
    TitanModeration moderation;
    TitanAuth auth;
    TitanMetrics metrics;           // Prometheus style metrics

    torch::Device device(torch::kCUDA);  // GPU

    TitanEngine engine(device);

    // ======== Chat API ========
    CROW_ROUTE(app, "/v1/chat").methods("POST"_method)
    ([&](const crow::request& req){

        metrics.inc("titancore_requests_total");
        auto start_time = std::chrono::high_resolution_clock::now();

        // ----- Authorization -----
        auto key = req.get_header_value("Authorization");
        if (!auth.validate(key))
            return crow::response(401, "invalid api key");

        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "invalid json");

        std::string prompt = body["prompt"].s();
        std::string client = req.remote_ip_address;

        // ----- Rate limiting -----
        if (!rate.allow(client))
            return crow::response(429, "rate limited");

        // ----- Jailbreak detection -----
        if (!jailbreak.is_safe(prompt))
            return crow::response(403, "prompt rejected");

        // ----- Embedding -----
        auto emb = engine.embed(prompt);

        // ----- Moderation -----
        if (!moderation.is_allowed(prompt, emb))
            return crow::response(403, "content blocked");

        // ----- Generate response -----
        auto output = engine.generate(prompt);

        // ----- Output moderation -----
        auto out_emb = engine.embed(output);
        if (!moderation.is_allowed(output, out_emb))
            return crow::response(403, "output blocked");

        // ----- Update Metrics -----
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        metrics.set("titancore_latency_ms", duration_ms);

        core_metrics.update_tokens(output.size());
        core_metrics.update_gpu_vram(engine.gpu_vram_usage());
        core_metrics.update_kv_cache(engine.kv_cache_usage());

        // ----- Return response -----
        json res;
        res["response"] = output;
        return crow::response{res};
    });

    // ======== Metrics endpoint ========
    CROW_ROUTE(app, "/metrics")([&](){
        return metrics.render();
    });

    std::cout << "[API] TitanCore listening on 0.0.0.0:8080\n";

    app.port(8080).multithreaded().run();
}
