#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
====================================================
 TITANCORE FLASH ATTENTION (Simplified CUDA Kernel)
====================================================
*/

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "Tensor must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// ------------------------------------------------
// CUDA Kernel
// ------------------------------------------------

__global__ void flash_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int B, int T, int C)
{
    int b = blockIdx.x;
    int t = threadIdx.x;

    if (t >= T) return;

    Q += b*T*C;
    K += b*T*C;
    V += b*T*C;
    O += b*T*C;

    // Shared memory for scores
    extern __shared__ float scores[];

    // Compute attention
    for(int i=0;i<T;i++) {

        float dot = 0.f;

        for(int c=0;c<C;c++)
            dot += __half2float(Q[t*C+c]) * __half2float(K[i*C+c]);

        scores[i] = dot;
    }

    // Softmax
    float maxv=-1e9;
    for(int i=0;i<T;i++) maxv = max(maxv, scores[i]);

    float sum=0;
    for(int i=0;i<T;i++){
        scores[i]=expf(scores[i]-maxv);
        sum+=scores[i];
    }

    // Weighted value sum
    for(int c=0;c<C;c++){
        float out=0;
        for(int i=0;i<T;i++)
            out+=scores[i]/sum * __half2float(V[i*C+c]);

        O[t*C+c]=__float2half(out);
    }
}


// ------------------------------------------------
// C++ Interface
// ------------------------------------------------

torch::Tensor flash_attention(torch::Tensor Q,
                              torch::Tensor K,
                              torch::Tensor V) {

    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    auto B = Q.size(0);
    auto T = Q.size(1);
    auto C = Q.size(2);

    auto O = torch::zeros_like(Q);

    const int threads = T;
    const dim3 blocks(B);

    flash_attention_kernel<<<blocks, threads, T*sizeof(float)>>>(
        (half*)Q.data_ptr<at::Half>(),
        (half*)K.data_ptr<at::Half>(),
        (half*)V.data_ptr<at::Half>(),
        (half*)O.data_ptr<at::Half>(),
        B,T,C);

    return O;
}


// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention, "TitanCore FlashAttention");
}
