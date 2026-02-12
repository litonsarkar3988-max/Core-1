#pragma once
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

/*
=====================================================
 TITANCORE KV CACHE
 Per session / per layer Key-Value memory
=====================================================
*/

struct KVLayerCache {
    torch::Tensor key;
    torch::Tensor value;
};

class KVCacheManager {

private:

    // session_id → layers
    std::unordered_map<int64_t, std::vector<KVLayerCache>> sessions;

public:

    void reset(int64_t session_id) {
        sessions.erase(session_id);
    }

    bool has_session(int64_t session_id) {
        return sessions.find(session_id) != sessions.end();
    }

    KVLayerCache& get_layer(int64_t session_id, int layer) {

        auto& layers = sessions[session_id];

        if (layers.size() <= layer)
            layers.resize(layer + 1);

        return layers[layer];
    }

    /*
        Append new K/V tensors

        k,v shape:
        [B, heads, T, dim]
    */

    void append(
        int64_t session_id,
        int layer,
        torch::Tensor k,
        torch::Tensor v
    ) {

        auto& cache = get_layer(session_id, layer);

        if (!cache.key.defined()) {
            cache.key = k;
            cache.value = v;
            return;
        }

        // concatenate on time dimension (dim=2)
        cache.key = torch::cat({cache.key, k}, 2);
        cache.value = torch::cat({cache.value, v}, 2);
    }

    /*
        Fetch cached tensors
    */

    bool fetch(
        int64_t session_id,
        int layer,
        torch::Tensor& k,
        torch::Tensor& v
    ) {

        if (!has_session(session_id)) return false;

        auto& layers = sessions[session_id];
        if (layer >= layers.size()) return false;

        auto& cache = layers[layer];
        if (!cache.key.defined()) return false;

        k = cache.key;
        v = cache.value;

        return true;
    }

    /*
        Current sequence length
    */

    int64_t seq_len(int64_t session_id, int layer) {
        auto& cache = get_layer(session_id, layer);
        if (!cache.key.defined()) return 0;
        return cache.key.size(2);
    }
};
