#include <torch/torch.h>
#include <iostream>

/*
========================================
 TITANCORE INT4 QUANT ENGINE
 2 weights packed per byte
========================================
*/

struct TitanINT4 {

    struct QTensor4 {
        torch::Tensor packed;   // uint8 (2x int4 inside)
        float scale;
        int64_t original_size;
    };

    // FP16/FP32 → INT4 ([-8,7])
    static QTensor4 quantize(torch::Tensor w) {

        auto max = w.abs().max().item<float>();
        float scale = max / 7.0f;

        auto q = torch::round(w / scale).clamp(-8, 7).to(torch::kInt8);

        auto flat = q.flatten();
        int64_t N = flat.numel();

        // pad if odd
        if (N % 2 != 0) {
            flat = torch::cat({flat, torch::zeros({1}, flat.options())});
            N++;
        }

        auto a = flat.slice(0, 0, N, 2);
        auto b = flat.slice(0, 1, N, 2);

        // pack: low nibble + high nibble
        auto packed = ((a & 0x0F) | ((b & 0x0F) << 4)).to(torch::kUInt8);

        return {packed, scale, w.numel()};
    }

    // INT4 → FP16
    static torch::Tensor dequantize(const QTensor4& qt) {

        auto bytes = qt.packed.to(torch::kInt16);

        auto lo = (bytes & 0x0F);
        auto hi = ((bytes >> 4) & 0x0F);

        lo = torch::where(lo > 7, lo - 16, lo);
        hi = torch::where(hi > 7, hi - 16, hi);

        auto full = torch::stack({lo, hi}, 1).flatten();

        full = full.slice(0, 0, qt.original_size);

        return full.to(torch::kFloat16) * qt.scale;
    }

    // INT4 Linear (dequant → GEMM)
    static torch::Tensor linear(
        torch::Tensor x,
        const QTensor4& w,
        torch::Tensor bias
    ) {
        auto weight = dequantize(w);
        return torch::linear(x, weight, bias);
    }
};
