#include <torch/extension.h>
#include <torch/torch.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/*
====================================================
  TITANCORE: NCCL ULTRA-PARALLEL ENGINE
====================================================
  Handles Tensor Parallelism (All-Reduce) and
  Pipeline Parallelism (Send/Recv) for 1T+ Models.
====================================================
*/

#define NCCL_CHECK(cmd) do {                         \
    ncclResult_t res = cmd;                          \
    if (res != ncclSuccess) {                        \
        printf("NCCL Error at %s:%d - %d\n",         \
               __FILE__, __LINE__, res);             \
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

class TitanNCCLManager {
private:
    ncclComm_t comm;
    ncclUniqueId id;
    int rank;
    int world_size;
    cudaStream_t stream;

public:
    TitanNCCLManager(int r, int ws) : rank(r), world_size(ws) {
        cudaStreamCreate(&stream);

        if (rank == 0) {
            ncclGetUniqueId(&id);
        }
        
        // Broadcast ID to all ranks
        torch::distributed::broadcast(&id, 0);
        
        NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
    }

    ~TitanNCCLManager() {
        ncclCommDestroy(comm);
        cudaStreamDestroy(stream);
    }

    // --------------------------------------------------
    // TENSOR PARALLELISM: ALL-REDUCE
    // --------------------------------------------------
    // 
    torch::Tensor all_reduce(torch::Tensor tensor) {
        auto options = tensor.options();
        ncclDataType_t dtype = (tensor.scalar_type() == torch::kFloat16) ? ncclFloat16 : ncclFloat32;

        NCCL_CHECK(ncclAllReduce(
            tensor.data_ptr(),
            tensor.data_ptr(),
            tensor.numel(),
            dtype,
            ncclSum,
            comm,
            stream
        ));
        
        return tensor;
    }

    // --------------------------------------------------
    // PIPELINE PARALLELISM: SEND / RECV
    // --------------------------------------------------
    // 
    void send(torch::Tensor tensor, int peer_rank) {
        ncclDataType_t dtype = (tensor.scalar_type() == torch::kFloat16) ? ncclFloat16 : ncclFloat32;
        
        NCCL_CHECK(ncclSend(
            tensor.data_ptr(),
            tensor.numel(),
            dtype,
            peer_rank,
            comm,
            stream
        ));
    }

    void recv(torch::Tensor& tensor, int peer_rank) {
        ncclDataType_t dtype = (tensor.scalar_type() == torch::kFloat16) ? ncclFloat16 : ncclFloat32;
        
        NCCL_CHECK(ncclRecv(
            tensor.data_ptr(),
            tensor.numel(),
            dtype,
            peer_rank,
            comm,
            stream
        ));
    }

    void sync() {
        cudaStreamSynchronize(stream);
    }
};

// --------------------------------------------------
// GLOBAL ENGINE INSTANCE
// --------------------------------------------------
static TitanNCCLManager* global_nccl = nullptr;

void init_nccl(int rank, int world_size) {
    if (!global_nccl) {
        global_nccl = new TitanNCCLManager(rank, world_size);
    }
}

 TitanNCCLManager* get_nccl() {
    return global_nccl;
}
