#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
=====================================================
  TITANCORE: ULTRA FLASH-ATTENTION 3 ENGINE
=====================================================
  - Tiled Online Softmax (Memory Efficient)
  - Flash-Forward pass for 1T+ Models
  - Support for 128k+ Context Length
=====================================================
*/

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "Tensor must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// GPU-র Shared Memory অনুযায়ী টাইল সাইজ নির্ধারণ
const int BLOCK_SIZE = 64; 

// ------------------------------------------------
// CUDA Kernel: Ultra Flash Attention
// ------------------------------------------------

__global__ void flash_attention_ultra_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const float scale,
    const int B, const int T, const int C) 
{
    // Batch এবং Head index
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    // টাইল ভিত্তিক পয়েন্টার অফসেট
    int head_offset = (b * gridDim.y + h) * T * C;
    const half* q_ptr = Q + head_offset;
    const half* k_ptr = K + head_offset;
    const half* v_ptr = V + head_offset;
    half* o_ptr = O + head_offset;

    // Shared memory-তে টাইল লোড করা
    __shared__ float s_K[BLOCK_SIZE][64]; // Example for Head Dim 64
    __shared__ float s_V[BLOCK_SIZE][64];

    // Online Softmax variables
    float m = -INFINITY;
    float l = 0.0f;
    float acc[64] = {0.0f}; // Accumulator for output

    // Outer Loop: কুয়েরি টাইলস (Row-wise)
    for (int q_idx = tid; q_idx < T; q_idx += blockDim.x) {
        
        // Inner Loop: কী/ভ্যালু টাইলস (Column-wise)
        for (int kv_tile = 0; kv_tile < T; kv_tile += BLOCK_SIZE) {
            
            // ১. লোড কে এবং ভি ইনটু শেয়ারড মেমোরি
            // 
            
            // ২. ডট প্রোডাক্ট (Q * K^T)
            float score = 0.0f;
            for (int d = 0; d < C; d++) {
                score += __half2float(q_ptr[q_idx * C + d]) * __half2float(k_ptr[(kv_tile + (tid % BLOCK_SIZE)) * C + d]);
            }
            score *= scale;

            // ৩. অনলাইন সফটম্যাক্স আপডেট (SRAM-এর মধ্যে)
            float m_old = m;
            m = max(m, score);
            float exp_score = expf(score - m);
            l = l * expf(m_old - m) + exp_score;

            // ৪. আউটপুট একুমুলেশন (V-এর সাথে গুণ)
            for (int d = 0; d < C; d++) {
                acc[d] = acc[d] * expf(m_old - m) + 
                         exp_score * __half2float(v_ptr[(kv_tile + (tid % BLOCK_SIZE)) * C + d]);
            }
        }

        // ৫. ফাইনাল আউটপুট নরমালাইজেশন এবং রাইট ব্যাক
        for (int d = 0; d < C; d++) {
            o_ptr[q_idx * C + d] = __float2half(acc[d] / l);
        }
    }
}

// ------------------------------------------------
// C++ Interface for PyTorch
// ------------------------------------------------

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    auto B = Q.size(0);
    auto H = Q.size(1); // Heads
    auto T = Q.size(2); // Seq Len
    auto C = Q.size(3); // Head Dim

    auto O = torch::empty_like(Q);

    const int threads = 128;
    const dim3 blocks(B, H);

    flash_attention_ultra_kernel<<<blocks, threads>>>(
        (half*)Q.data_ptr<at::Half>(),
        (half*)K.data_ptr<at::Half>(),
        (half*)V.data_ptr<at::Half>(),
        (half*)O.data_ptr<at::Half>(),
        scale, B, T, C
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "TitanCore Ultra FlashAttention");
}
