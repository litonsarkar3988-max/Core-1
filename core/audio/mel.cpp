#include <torch/torch.h>
#include <cmath>

/*
====================================================
 TITANCORE MEL SPECTROGRAM ENGINE
 Raw Audio → Log-Mel Features
====================================================
*/

constexpr int SAMPLE_RATE = 16000;
constexpr int N_FFT = 400;
constexpr int HOP = 160;
constexpr int N_MELS = 80;

/* ---------------- Window ---------------- */

torch::Tensor hann_window() {
    return torch::hann_window(N_FFT, torch::kFloat32);
}

/* ---------------- Mel Scale ---------------- */

float hz_to_mel(float hz) {
    return 2595.f * log10f(1.f + hz / 700.f);
}

float mel_to_hz(float mel) {
    return 700.f * (powf(10.f, mel / 2595.f) - 1.f);
}

/* ---------------- Mel Filter Bank ---------------- */

torch::Tensor mel_filter() {

    int n_freq = N_FFT/2 + 1;

    auto fb = torch::zeros({N_MELS, n_freq});

    float mel_min = hz_to_mel(0);
    float mel_max = hz_to_mel(SAMPLE_RATE / 2);

    std::vector<float> mel_pts(N_MELS+2);

    for(int i=0;i<N_MELS+2;i++)
        mel_pts[i] = mel_min + (mel_max-mel_min)*i/(N_MELS+1);

    std::vector<int> bins(N_MELS+2);

    for(int i=0;i<N_MELS+2;i++)
        bins[i] = floor((N_FFT+1)*mel_to_hz(mel_pts[i])/SAMPLE_RATE);

    for(int m=1;m<=N_MELS;m++) {
        for(int k=bins[m-1];k<bins[m];k++)
            fb[m-1][k] = (k - bins[m-1]) / float(bins[m]-bins[m-1]);

        for(int k=bins[m];k<bins[m+1];k++)
            fb[m-1][k] = (bins[m+1]-k) / float(bins[m+1]-bins[m]);
    }

    return fb;
}

/* ---------------- Log Mel Spectrogram ---------------- */

torch::Tensor log_mel(torch::Tensor audio) {

    auto window = hann_window().to(audio.device());

    auto stft = torch::stft(
        audio,
        N_FFT,
        HOP,
        N_FFT,
        window,
        true,
        "reflect",
        false,
        true
    );

    auto mag = torch::sqrt(stft.pow(2).sum(-1));

    auto mel_fb = mel_filter().to(audio.device());

    auto mel = torch::matmul(mel_fb, mag);

    mel = torch::clamp(mel, 1e-10);

    return torch::log10(mel);
}
