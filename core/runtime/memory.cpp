#include <torch/torch.h>
#include <unordered_map>
#include <mutex>

/*
=====================================
 TITANCORE GPU MEMORY MANAGER
 VRAM POOL + KV CACHE CONTROLLER
=====================================
*/

struct MemoryBlock {
    torch::Tensor tensor;
    size_t bytes;
    bool used = false;
};

class TitanMemory {

private:

    std::vector<MemoryBlock> pool;
    std::mutex lock;

    size_t total = 0;
    size_t limit;

public:

    TitanMemory(size_t vram_limit_bytes)
        : limit(vram_limit_bytes) {}

    /* allocate tensor from pool */

    torch::Tensor alloc(std::vector<int64_t> shape,
                        torch::ScalarType dtype,
                        torch::Device device) {

        std::unique_lock<std::mutex> guard(lock);

        size_t need = torch::zeros(shape,dtype).numel() * torch::elementSize(dtype);

        /* reuse block */

        for(auto& b:pool) {
            if(!b.used && b.bytes >= need) {
                b.used = true;
                return b.tensor.view(shape);
            }
        }

        if(total + need > limit) {
            evict();
        }

        auto t = torch::zeros(shape,torch::TensorOptions().dtype(dtype).device(device));

        pool.push_back({t,need,true});
        total += need;

        return t;
    }

    /* free */

    void release(torch::Tensor t) {

        std::unique_lock<std::mutex> guard(lock);

        for(auto& b:pool) {
            if(b.tensor.data_ptr() == t.data_ptr()) {
                b.used = false;
                return;
            }
        }
    }

    /* simple LRU eviction */

    void evict() {

        for(auto& b:pool) {
            if(!b.used) {
                total -= b.bytes;
                b.tensor.reset();
                b.bytes = 0;
                return;
            }
        }

        TORCH_CHECK(false,"VRAM exhausted");
    }
};
