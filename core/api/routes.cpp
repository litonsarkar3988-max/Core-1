#include "crow_all.h"

#include "../runtime/engine.cpp"
#include "../safety/jailbreak.cpp"
#include "../safety/moderation.cpp"
#include "../safety/rate_limit.cpp"

using json = crow::json::wvalue;

/*
=========================================
 TITANCORE ROUTES
 All API endpoints live here
=========================================
*/

void register_routes(
    crow::SimpleApp& app,
    TitanEngine& engine,
    TitanRateLimiter& rate,
    TitanJailbreak& jailbreak,
    TitanModeration& moderation
) {

    // ---------------------------
    // CHAT ENDPOINT
    // ---------------------------
    CROW_ROUTE(app, "/v1/chat").methods("POST"_method)
    ([&](const crow::request& req){

        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400);

        std::string prompt = body["prompt"].s();
        std::string client = req.remote_ip_address;

        if (!rate.allow(client))
            return crow::response(429, "rate limited");

        if (!jailbreak.is_safe(prompt))
            return crow::response(403, "jailbreak detected");

        auto emb = engine.embed(prompt);

        if (!moderation.is_allowed(prompt, emb))
            return crow::response(403, "input blocked");

        auto reply = engine.generate(prompt);

        auto out_emb = engine.embed(reply);
        if (!moderation.is_allowed(reply, out_emb))
            return crow::response(403, "output blocked");

        json res;
        res["response"] = reply;

        return crow::response{res};
    });

    // ---------------------------
    // EMBEDDING ENDPOINT
    // ---------------------------
    CROW_ROUTE(app, "/v1/embed").methods("POST"_method)
    ([&](const crow::request& req){

        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400);

        std::string text = body["text"].s();

        auto vec = engine.embed(text);

        json out;
        for (int i = 0; i < vec.size(0); ++i)
            out["embedding"][i] = vec[i].item<float>();

        return crow::response{out};
    });

    // ---------------------------
    // HEALTH CHECK
    // ---------------------------
    CROW_ROUTE(app, "/health")
    ([](){
        return crow::response(200, "ok");
    });
}
