#include "crow_all.h"          // Crow single-header web framework
#include <torch/torch.h>

#include "../runtime/engine.cpp"
#include "../safety/jailbreak.cpp"
#include "../safety/moderation.cpp"
#include "../safety/rate_limit.cpp"
#include "../retrieval/embedder.cpp"

using json = crow::json::wvalue;

/*
=========================================
 TITANCORE INFERENCE SERVER
=========================================
*/

int main() {

    crow::SimpleApp app;

    TitanRateLimiter rate;
    TitanJailbreak jailbreak;
    TitanModeration moderation;

    torch::Device device(torch::kCUDA);

    TitanEngine engine(device);

    CROW_ROUTE(app, "/v1/chat").methods("POST"_method)
    ([&](const crow::request& req){

        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400);

        std::string prompt = body["prompt"].s();

        std::string client = req.remote_ip_address;

        // 1. Rate limit
        if (!rate.allow(client))
            return crow::response(429, "rate limited");

        // 2. Jailbreak detection
        if (!jailbreak.is_safe(prompt))
            return crow::response(403, "prompt rejected");

        // 3. Embed prompt
        auto emb = engine.embed(prompt);

        // 4. Moderation
        if (!moderation.is_allowed(prompt, emb))
            return crow::response(403, "content blocked");

        // 5. Generate response
        auto output = engine.generate(prompt);

        // 6. Output moderation
        auto out_emb = engine.embed(output);
        if (!moderation.is_allowed(output, out_emb))
            return crow::response(403, "output blocked");

        json res;
        res["response"] = output;

        return crow::response{res};
    });

    std::cout << "[API] TitanCore listening on 0.0.0.0:8080\n";

    app.port(8080).multithreaded().run();
}
