TitanCore 🚀





TitanCore is a lightweight, high-performance modular AI inference engine designed for multimodal capabilities (Text, Vision, and Audio).

> 💡 Fun Fact:
This engine was entirely architected and developed on a mobile device.




---

⚠️ Project Status & Tokenization

Training Status:
Structural framework only. NOT fully trained.

Tokenization:
Custom vocab.json + merges.txt
~400,000 tokens (4 Lakh)

Note:
Specialized fast GPU inference (not trillion-token model)



---

🏗️ System Architecture

titancore/
├── core/
├── distributed/
├── quant/
├── retrieval/
├── safety/
├── api/
└── tools/


---

🛠️ Technical Specifications

Development Platform : Android / Termux
Vocabulary Size      : ~400,000
Model Format         : GGUF (titancore.gguf)
Core Inference       : C++17 Transformer
Acceleration         : FlashAttention CUDA
Precision            : FP32 / FP16 / INT8 / INT4


---

🚀 Getting Started

Prerequisites:

CMake 3.18+

CUDA Toolkit

C++17 Compiler

NVIDIA GPU (CPU NOT supported)


Build:

git clone https://github.com/litonsarkar3988-max/Core-1
cd titancore

mkdir build && cd build
cmake ..
make -j$(nproc)

Run (GPU Only):

./main --model ../weights/titancore.gguf --config ../configs/gpt4o.yaml


---

🛡️ Safety

Moderation + jailbreak detection included.


---

⚠️ Important

Inference only

Requires pretrained weights

GPU mandatory

Tokenizer ~400k

Experimental research project



---

👥 Author

Rahul Sarkar — India 🇮🇳
Personal AI research project developed entirely on mobile.
