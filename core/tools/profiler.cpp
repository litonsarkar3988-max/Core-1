#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <chrono>

#define CHECK_CUDA(x) if((x)!=cudaSuccess){printf("CUDA error\n");exit(0);}

// =====================================
// Simple CUDA timer
// =====================================
struct CUDATimer {
    cudaEvent_t start, stop;

    CUDATimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    void tic() { cudaEventRecord(start); }
    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms,start,stop);
        return ms;
    }
};

// =====================================
// Global profiling registry
// =====================================
std::unordered_map<std::string,double> registry;

void profile(const std::string& tag, std::function<void()> fn) {
    CUDATimer t;
    t.tic();
    fn();
    float ms = t.toc();
    registry[tag] += ms;
}

// =====================================
// VRAM monitor
// =====================================
void print_memory() {
    size_t free,total;
    cudaMemGetInfo(&free,&total);
    std::cout << "[VRAM] Used: "
              << (total-free)/1024/1024
              << " MB\n";
}

// =====================================
// Example hook usage
// =====================================
void dump() {
    std::cout << "\n======= PROFILER =======\n";
    for(auto& kv: registry)
        std::cout << kv.first << ": " << kv.second << " ms\n";
    print_memory();
    std::cout << "========================\n";
}

// =====================================
// Example integration
// =====================================
// profile("attention",[&](){
//     attention_forward(...);
// });
//
// profile("mlp",[&](){
//     mlp_forward(...);
// });

int main() {

    torch::Device device(torch::kCUDA);

    auto x = torch::randn({4,1024,2048}).to(device);

    profile("linear", [&](){
        auto y = torch::matmul(x,x.transpose(1,2));
        torch::cuda::synchronize();
    });

    profile("relu", [&](){
        auto y = torch::relu(x);
        torch::cuda::synchronize();
    });

    dump();
}
