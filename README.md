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

Core-1/
├── core/                        # Neural brain
│   ├── model/
│   │   ├── gpt.cpp             # Transformer decoder
│   │   ├── attention.cu        # FlashAttention CUDA kernel
│   │   ├── block.cpp           # Transformer blocks
│   │   ├── embedding.cpp       # Token + position embedding
│   │   ├── kv_cache.cpp        # KV memory
│   │   └── sampler.cpp         # top-k / top-p / temp
│   │
│   ├── tokenizer/
│   │   ├── sentencepiece.cpp
│   │   ├── vocab.json
│   │   └── merges.txt
│   │
│   ├── vision/                 # multimodal
│   │   ├── vit.cpp             # vision transformer
│   │   └── clip.cpp
│   │
│   ├── audio/
│   │   ├── whisper.cpp
│   │   └── mel.cpp
│   │
│   └── runtime/
│       ├── engine.cpp          # inference engine
│       ├── scheduler.cpp       # batching
│       └── memory.cpp          # VRAM manager
│
├── distributed/
│   ├── nccl.cpp                # tensor parallel
│   ├── fsdp.cpp                # shard weights
│   └── mpi.cpp
│
├── quant/
│   ├── int8.cpp
│   ├── int4.cpp
│   └── fp8.cpp
│
├── retrieval/                  # RAG
│   ├── faiss.cpp
│   ├── embedder.cpp
│   └── loader.cpp
│
├── safety/
│   ├── moderation.cpp
│   ├── jailbreak.cpp
│   └── rate_limit.cpp
│
├── api/
│   ├── server.cpp              # REST / gRPC
│   ├── routes.cpp
│   └── auth.cpp
│
├── monitoring/
│   ├── prometheus.cpp
│   └── metrics.cpp
│
├── tools/
│   ├── convert_weights.py
│   ├── benchmark.cpp
│   └── profiler.cpp
│
├── configs/
│   ├── gpt4o.yaml
│   ├── cluster.yaml
│   └── safety.yaml
│
├── weights/
│   └── titancore.gguf
│
├── main.cpp                    # system bootstrap
└── CMakeLists.txt


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
