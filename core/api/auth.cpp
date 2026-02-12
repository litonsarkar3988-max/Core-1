#include <unordered_map>
#include <string>
#include <iostream>

/*
=========================================
 TITANCORE AUTH MODULE
 API Key validation + tiering
=========================================
*/

struct ApiKeyInfo {
    std::string owner;
    int tier;          // 0 = free, 1 = pro, 2 = enterprise
    bool active;
};

class TitanAuth {

private:

    // Normally loaded from DB / Redis
    std::unordered_map<std::string, ApiKeyInfo> api_keys = {
        {"demo-key-123", {"rahul", 2, true}},
        {"free-key-001", {"guest", 0, true}}
    };

public:

    bool validate(const std::string& key) {

        if (!api_keys.count(key)) {
            std::cout << "[Auth] Unknown key\n";
            return false;
        }

        if (!api_keys[key].active) {
            std::cout << "[Auth] Revoked key\n";
            return false;
        }

        return true;
    }

    ApiKeyInfo info(const std::string& key) {
        return api_keys[key];
    }

};
