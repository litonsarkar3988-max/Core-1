#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "engine.cpp"

/*
=========================================
 TITANCORE SCHEDULER
 Dynamic Microbatch Engine
=========================================
*/

struct InferRequest {
    int session;
    torch::Tensor tokens;
    std::function<void(torch::Tensor)> callback;
};

class TitanScheduler {

private:

    std::queue<InferRequest> queue;
    std::mutex lock;
    std::condition_variable cv;

    TitanEngine* engine;

    int max_batch = 16;
    int max_seq = 512;

public:

    TitanScheduler(TitanEngine* e):engine(e){
        std::thread(&TitanScheduler::loop,this).detach();
    }

    /* push request */

    void submit(torch::Tensor tokens,
                std::function<void(torch::Tensor)> cb) {

        static int sid = 0;

        std::unique_lock<std::mutex> guard(lock);

        queue.push({
            sid++,
            tokens,
            cb
        });

        cv.notify_one();
    }

    /* batching loop */

    void loop() {

        while(true) {

            std::vector<InferRequest> batch;

            {
                std::unique_lock<std::mutex> guard(lock);

                cv.wait(guard,[&]{return !queue.empty();});

                while(!queue.empty() && batch.size() < max_batch) {
                    batch.push_back(queue.front());
                    queue.pop();
                }
            }

            if(batch.empty()) continue;

            /* pad */

            int maxlen = 0;
            for(auto& r:batch)
                maxlen = std::max(maxlen,(int)r.tokens.size(1));

            std::vector<torch::Tensor> padded;

            for(auto& r:batch) {

                auto t = r.tokens;

                if(t.size(1) < maxlen) {
                    auto pad = torch::zeros({1,maxlen-t.size(1)},torch::kLong);
                    t = torch::cat({t,pad},1);
                }

                padded.push_back(t);
            }

            auto input = torch::cat(padded,0).to(engine->device);

            auto logits = engine->gpt->forward(input,&engine->kv,0);

            auto split = logits.split(1,0);

            for(int i=0;i<batch.size();i++)
                batch[i].callback(split[i]);
        }
    }
};
