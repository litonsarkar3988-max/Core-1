#include <torch/torch.h>
#include <iostream>

/*
========================================
 TITANCORE INT8 QUANTIZATION ENGINE
 Symmetric per-tensor quant
========================================
*/

struct TitanINT8 {

    struct QTensor {
        torch::Tensor qweight;   // int8
        float scale;
    };

    /* FP16/FP32 → INT8 */

    static QTensor quantize(torch::Tensor weight) {

        auto max = weight.abs().max().item<float>();
        float scale = max / 127.0f;

        auto q = torch::clamp(
            torch::round(weight / scale),
            -128, 127
        ).to(torch::kInt8);

        return {q, scale};
    }

    /* INT8 → FP16 */

    static torch::Tensor dequantize(const QTensor& qt) {
        return qt.qweight.to(torch::kFloat16) * qt.scale;
    }

    /* INT8 GEMM */

    static torch::Tensor linear(
        torch::Tensor input,
        const QTensor& weight,
        torch::Tensor bias
    ) {
        auto w = dequantize(weight);
        return torch::linear(input, w, bias);
    }
};
