#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

// NNUE architecture: HalfKP(feature set) -> 256 -> 32 -> 32 -> 1
constexpr int NNUE_FT_INPUTS  = 40960;  // HalfKP features: 64*10*64
constexpr int NNUE_FT_OUTPUTS = 256;
constexpr int NNUE_L1_SIZE    = 32;
constexpr int NNUE_L2_SIZE    = 32;
constexpr int NNUE_L3_SIZE    = 1;

constexpr int NNUE_SCALE = 400;
constexpr int NNUE_QA    = 255;   // Input quantization
constexpr int NNUE_QB    = 64;    // Hidden quantization

struct NNUEAccumulator {
    int16_t values[NNUE_FT_OUTPUTS];
};

class NNUE {
public:
    NNUE();
    ~NNUE();

    bool loadWeights(const std::string& filename);
    bool isLoaded() const { return weights_loaded_; }

    // Refresh accumulator for a set of feature indices
    void refreshAccumulator(NNUEAccumulator& acc,
                            const uint32_t* features, int count) const;

    // Forward pass from accumulator -> centipawn score
    int forward(const NNUEAccumulator& acc) const;

    // Evaluate position (returns centipawn score from side-to-move perspective)
    int evaluate(const uint32_t* whiteFeatures, int wCount,
                 const uint32_t* blackFeatures, int bCount) const;

    // Extract HalfKP feature indices from piece bitboards
    static void extractFeatures(const uint64_t* pieceBB,
                                uint32_t* whiteFeat, int& wCount,
                                uint32_t* blackFeat, int& bCount);

    // Training API (calls trainInit first, then trainStep for each position)
    bool trainInit();
    float predict(const uint64_t* pieceBB) const;
    void trainStep(const uint64_t* pieceBB, float targetScore, float lr);
    bool saveWeights(const std::string& filename) const;

private:
    int16_t* ft_weights_ = nullptr;
    int16_t* ft_biases_  = nullptr;
    int8_t*  l1_weights_ = nullptr;
    int32_t* l1_biases_  = nullptr;
    int8_t*  l2_weights_ = nullptr;
    int32_t* l2_biases_  = nullptr;
    int8_t*  l3_weights_ = nullptr;
    int32_t* l3_biases_  = nullptr;
    bool     weights_loaded_ = false;

    // Float copies for training
    float* ft_weights_f_ = nullptr;
    float* ft_biases_f_  = nullptr;
    float* l1_weights_f_ = nullptr;
    float* l1_biases_f_  = nullptr;
    float* l2_weights_f_ = nullptr;
    float* l2_biases_f_  = nullptr;
    float* l3_weights_f_ = nullptr;
    float* l3_biases_f_  = nullptr;

    // Adam optimizer state buffers
    float* ft_m_ = nullptr;  float* ft_v_ = nullptr;
    float* ft_b_m_ = nullptr; float* ft_b_v_ = nullptr;
    float* l1_m_ = nullptr;  float* l1_v_ = nullptr;
    float* l1_b_m_ = nullptr; float* l1_b_v_ = nullptr;
    float* l2_m_ = nullptr;  float* l2_v_ = nullptr;
    float* l2_b_m_ = nullptr; float* l2_b_v_ = nullptr;
    float* l3_m_ = nullptr;  float* l3_v_ = nullptr;
    float* l3_b_m_ = nullptr; float* l3_b_v_ = nullptr;
    int    adam_step_ = 0;

    // Scratch buffers for training (owned, reused across steps)
    float* scratch_ft_ = nullptr;
    float* scratch_l1_ = nullptr;
    float* scratch_l2_ = nullptr;

    static int halfKPIndex(int kingSq, int pType, int pSq) {
        return (kingSq * 10 + pType) * 64 + pSq;
    }
    static int clip(int x) { return std::max(0, std::min(127, x)); }
};
