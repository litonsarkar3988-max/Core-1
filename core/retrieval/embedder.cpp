#include <torch/torch.h>
#include <string>
#include "../core/model/gpt.cpp"

/*
=========================================
 TITANCORE EMBEDDER
 Text → Vector (RAG backbone)
=========================================
*/

struct TitanEmbedder {

    std::shared_ptr<TitancoreGPT> model;
    torch::Device device;

    TitanEmbedder(std::shared_ptr<TitancoreGPT> gpt, torch::Device dev)
        : model(gpt), device(dev) {}

    /*
      Input: token ids [1, T]
      Output: embedding vector [C]
    */
    torch::Tensor encode(torch::Tensor tokens) {

        tokens = tokens.to(device);

        torch::NoGradGuard no_grad;

        auto hidden = model->forward(tokens);  // [1,T,V]

        // Mean Pooling (GPT-4o style)
        auto emb = hidden.mean(1).squeeze(0);

        // L2 normalize (FAISS friendly)
        emb = torch::nn::functional::normalize(
            emb,
            torch::nn::functional::NormalizeFuncOptions().dim(0)
        );

        return emb.cpu();
    }

};
