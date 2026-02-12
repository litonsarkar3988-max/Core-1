#include <torch/torch.h>
#include "../core/config.h"

/*
====================================================
 TITANCORE ViT
 Vision Transformer Encoder
====================================================
*/

struct PatchEmbedding : torch::nn::Module {

    torch::nn::Conv2d proj{nullptr};
    int patch;

    PatchEmbedding(int img_size, int patch_size, int embd)
        : patch(patch_size) {

        proj = register_module(
            "proj",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, embd, patch).stride(patch)
            )
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        // [B,3,H,W] → [B,C,Hp,Wp]
        x = proj(x);

        // flatten patches
        x = x.flatten(2).transpose(1,2);
        return x;
    }
};

/* ---------------- Transformer Block ---------------- */

struct ViTBlock : torch::nn::Module {

    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::Sequential mlp{nullptr};

    ViTBlock(int dim, int heads) {

        ln1 = register_module("ln1", torch::nn::LayerNorm(dim));
        ln2 = register_module("ln2", torch::nn::LayerNorm(dim));

        attn = register_module(
            "attn",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(dim, heads)
            )
        );

        mlp = register_module("mlp", torch::nn::Sequential(
            torch::nn::Linear(dim, dim*4),
            torch::nn::GELU(),
            torch::nn::Linear(dim*4, dim)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {

        auto h = x;
        x = ln1(x);
        x = std::get<0>(attn(x,x,x));
        x = x + h;

        h = x;
        x = ln2(x);
        x = mlp->forward(x);
        x = x + h;

        return x;
    }
};

/* ---------------- Main ViT ---------------- */

struct TitanViT : torch::nn::Module {

    PatchEmbedding patcher{nullptr};
    torch::nn::Embedding pos_embed{nullptr};
    std::vector<ViTBlock> blocks;
    torch::nn::LayerNorm ln{nullptr};

    int num_patches;

    TitanViT(int img=224,int patch=16,int dim=768,int depth=12,int heads=12) {

        num_patches = (img/patch)*(img/patch);

        patcher = register_module("patcher", PatchEmbedding(img,patch,dim));

        pos_embed = register_module(
            "pos_embed",
            torch::nn::Embedding(num_patches,dim)
        );

        for(int i=0;i<depth;i++)
            blocks.push_back(register_module("block"+std::to_string(i),
                ViTBlock(dim,heads)));

        ln = register_module("ln", torch::nn::LayerNorm(dim));
    }

    torch::Tensor forward(torch::Tensor img) {

        auto x = patcher->forward(img);

        auto T = x.size(1);
        auto pos = torch::arange(0,T,torch::kLong).to(img.device());

        x = x + pos_embed(pos);

        for(auto& b:blocks)
            x = b.forward(x);

        return ln(x);
    }
};
