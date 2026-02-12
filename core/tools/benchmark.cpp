#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include "../core/model/gpt.cpp"
#include "engine.cpp"

using Clock = std::chrono::high_resolution_clock;

// -------------------------------------
// Simple percentile helper
// -------------------------------------
double percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * v.size());
    return v[idx];
}

// -------------------------------------
// GPU memory (mock)
// -------------------------------------
void print_gpu_memory() {
#ifdef USE_CUDA
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "VRAM: " << (total - free)/1024/1024 << "MB used\n";
#endif
}

// -------------------------------------
// Benchmark runner
// -------------------------------------
int main() {

    torch::Device device(torch::kCUDA);

    TitanConfig cfg;
    cfg.block_size = 1024;
    cfg.n_layer = 24;
    cfg.n_embd = 2048;
    cfg.vocab_size = 50257;

    auto model = std::make_shared<TitancoreGPT>(cfg);
    model->to(device);
    model->eval();

    KVCacheManager cache(cfg);

    const int B = 4;
    const int T = 128;
    const int runs = 100;

    std::vector<double> latencies;

    // Warmup
    std::cout << "[Warmup]\n";
    for(int i=0;i<10;i++){
        auto x = torch::randint(0,cfg.vocab_size,{B,T}).to(device);
        model->forward(x,&cache,0);
    }

    std::cout << "\n[Benchmark]\n";

    for(int i=0;i<runs;i++){

        auto x = torch::randint(0,cfg.vocab_size,{B,T}).to(device);

        auto start = Clock::now();
        model->forward(x,&cache,i);
        torch::cuda::synchronize();
        auto end = Clock::now();

        double ms = std::chrono::duration<double,std::milli>(end-start).count();
        latencies.push_back(ms);
    }

    double avg = std::accumulate(latencies.begin(),latencies.end(),0.0)/runs;

    std::cout << "\n==============================\n";
    std::cout << "Avg latency: " << avg << " ms\n";
    std::cout << "P50: " << percentile(latencies,0.50) << " ms\n";
    std::cout << "P95: " << percentile(latencies,0.95) << " ms\n";

    double tokens = runs * B * T;
    double tps = tokens / (std::accumulate(latencies.begin(),latencies.end(),0.0)/1000.0);

    std::cout << "Tokens/sec: " << tps << "\n";

    print_gpu_memory();

    std::cout << "==============================\n";

    return 0;
}
