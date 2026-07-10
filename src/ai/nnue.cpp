#include "nnue.hpp"
#include <cstdlib>
#include <cstring>
#include <cmath>

// Aligned allocation helpers
template<typename T> static T* alignedAlloc(size_t n) {
    void* p = nullptr;
    return (posix_memalign(&p, 64, n * sizeof(T)) == 0) ? static_cast<T*>(p) : nullptr;
}
template<typename T> static void alignedFree(T* p) { free(p); }

NNUE::NNUE() {}
NNUE::~NNUE() {
    alignedFree(ft_weights_); alignedFree(ft_biases_);
    alignedFree(l1_weights_); alignedFree(l1_biases_);
    alignedFree(l2_weights_); alignedFree(l2_biases_);
    alignedFree(l3_weights_); alignedFree(l3_biases_);
    alignedFree(ft_weights_f_); alignedFree(ft_biases_f_);
    alignedFree(l1_weights_f_); alignedFree(l1_biases_f_);
    alignedFree(l2_weights_f_); alignedFree(l2_biases_f_);
    alignedFree(l3_weights_f_); alignedFree(l3_biases_f_);
    alignedFree(scratch_ft_); alignedFree(scratch_l1_); alignedFree(scratch_l2_);
}

bool NNUE::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    char magic[4]; file.read(magic, 4);
    if (magic[0]!='N'||magic[1]!='N'||magic[2]!='U'||magic[3]!='E') return false;
    int32_t ver; file.read(reinterpret_cast<char*>(&ver), 4);
    if (ver != 1) return false;

    auto allocAll = [&]() -> bool {
        ft_weights_ = alignedAlloc<int16_t>(NNUE_FT_OUTPUTS * NNUE_FT_INPUTS);
        ft_biases_  = alignedAlloc<int16_t>(NNUE_FT_OUTPUTS);
        l1_weights_ = alignedAlloc<int8_t>(NNUE_L1_SIZE * NNUE_FT_OUTPUTS);
        l1_biases_  = alignedAlloc<int32_t>(NNUE_L1_SIZE);
        l2_weights_ = alignedAlloc<int8_t>(NNUE_L2_SIZE * NNUE_L1_SIZE);
        l2_biases_  = alignedAlloc<int32_t>(NNUE_L2_SIZE);
        l3_weights_ = alignedAlloc<int8_t>(NNUE_L3_SIZE * NNUE_L2_SIZE);
        l3_biases_  = alignedAlloc<int32_t>(NNUE_L3_SIZE);
        return ft_weights_ && ft_biases_ && l1_weights_ && l1_biases_
            && l2_weights_ && l2_biases_ && l3_weights_ && l3_biases_;
    };

    if (!allocAll()) return false;

    auto read = [&](void* buf, size_t sz) { file.read(static_cast<char*>(buf), sz); };

    read(ft_weights_, NNUE_FT_OUTPUTS * NNUE_FT_INPUTS * sizeof(int16_t));
    read(ft_biases_,  NNUE_FT_OUTPUTS * sizeof(int16_t));
    read(l1_weights_, NNUE_L1_SIZE * NNUE_FT_OUTPUTS * sizeof(int8_t));
    read(l1_biases_,  NNUE_L1_SIZE * sizeof(int32_t));
    read(l2_weights_, NNUE_L2_SIZE * NNUE_L1_SIZE * sizeof(int8_t));
    read(l2_biases_,  NNUE_L2_SIZE * sizeof(int32_t));
    read(l3_weights_, NNUE_L3_SIZE * NNUE_L2_SIZE * sizeof(int8_t));
    read(l3_biases_,  NNUE_L3_SIZE * sizeof(int32_t));

    weights_loaded_ = true;
    return true;
}

// HalfKP feature extraction
// pieceBB layout: 0=P_w,1=N_w,2=B_w,3=R_w,4=Q_w,5=K_w,6=P_b,7=N_b,8=B_b,9=R_b,10=Q_b,11=K_b
// HalfKP piece types: 0=P_w,1=N_w,2=B_w,3=R_w,4=Q_w,5=P_b,6=N_b,7=B_b,8=R_b,9=Q_b
void NNUE::extractFeatures(const uint64_t* pieceBB,
                           uint32_t* whiteFeat, int& wCount,
                           uint32_t* blackFeat, int& bCount) {
    wCount = bCount = 0;

    int wKing = pieceBB[5] ? __builtin_ctzll(pieceBB[5]) : -1;
    int bKing = pieceBB[11] ? __builtin_ctzll(pieceBB[11]) : -1;
    if (wKing < 0 || bKing < 0) return;

    // Piece type mapping from pieceBB index to NNUE piece type
    // pieceBB[0]=P_w -> type 0, pieceBB[1]=N_w -> type 1, ..., pieceBB[4]=Q_w -> type 4
    // pieceBB[6]=P_b -> type 5, pieceBB[7]=N_b -> type 6, ..., pieceBB[10]=Q_b -> type 9
    static const int pTypeMap[12] = {0,1,2,3,4,-1,5,6,7,8,9,-1};

    for (int i = 0; i < 12; ++i) {
        int pt = pTypeMap[i];
        if (pt < 0) continue; // skip kings
        uint64_t bb = pieceBB[i];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            whiteFeat[wCount++] = halfKPIndex(wKing, pt, sq);
            blackFeat[bCount++] = halfKPIndex(bKing, pt, sq);
        }
    }
}

void NNUE::refreshAccumulator(NNUEAccumulator& acc,
                              const uint32_t* features, int count) const {
    std::memcpy(acc.values, ft_biases_, NNUE_FT_OUTPUTS * sizeof(int16_t));
    for (int i = 0; i < count; ++i) {
        const int16_t* col = &ft_weights_[features[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            acc.values[j] += col[j];
    }
}

int NNUE::forward(const NNUEAccumulator& acc) const {
    // Hidden Layer 1: 256 -> 32
    int32_t l1[NNUE_L1_SIZE];
    for (int i = 0; i < NNUE_L1_SIZE; ++i) {
        int32_t sum = l1_biases_[i];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            sum += (int32_t)acc.values[j] * (int32_t)l1_weights_[i * NNUE_FT_OUTPUTS + j];
        l1[i] = clip(sum / NNUE_QB);
    }

    // Hidden Layer 2: 32 -> 32
    int32_t l2[NNUE_L2_SIZE];
    for (int i = 0; i < NNUE_L2_SIZE; ++i) {
        int32_t sum = l2_biases_[i];
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            sum += l1[j] * (int32_t)l2_weights_[i * NNUE_L1_SIZE + j];
        l2[i] = clip(sum / NNUE_QB);
    }

    // Output: 32 -> 1
    int32_t out = l3_biases_[0];
    for (int j = 0; j < NNUE_L2_SIZE; ++j)
        out += l2[j] * (int32_t)l3_weights_[j];

    // Scale to centipawns (multiply by SCALE and divide by QA*QB)
    return out * NNUE_SCALE / (NNUE_QA * NNUE_QB);
}

int NNUE::evaluate(const uint32_t* whiteFeat, int wCount,
                   const uint32_t* blackFeat, int bCount) const {
    if (!weights_loaded_) return 0;

    NNUEAccumulator acc;
    refreshAccumulator(acc, whiteFeat, wCount);
    return forward(acc);
}

float NNUE::predict(const uint64_t* pieceBB) const {
    if (!ft_weights_f_) return 0.0f;

    uint32_t whiteFeat[32], blackFeat[32];
    int wc, bc;
    extractFeatures(pieceBB, whiteFeat, wc, blackFeat, bc);

    float ft[NNUE_FT_OUTPUTS];
    std::memcpy(ft, ft_biases_f_, NNUE_FT_OUTPUTS * sizeof(float));
    for (int i = 0; i < wc; ++i) {
        const float* col = &ft_weights_f_[whiteFeat[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            ft[j] += col[j];
    }

    float l1[NNUE_L1_SIZE];
    for (int i = 0; i < NNUE_L1_SIZE; ++i) {
        float sum = l1_biases_f_[i];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            sum += ft[j] * l1_weights_f_[i * NNUE_FT_OUTPUTS + j];
        l1[i] = std::max(0.0f, std::min(127.0f, sum / NNUE_QB));
    }

    float l2[NNUE_L2_SIZE];
    for (int i = 0; i < NNUE_L2_SIZE; ++i) {
        float sum = l2_biases_f_[i];
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            sum += l1[j] * l2_weights_f_[i * NNUE_L1_SIZE + j];
        l2[i] = std::max(0.0f, std::min(127.0f, sum / NNUE_QB));
    }

    float out = l3_biases_f_[0];
    for (int j = 0; j < NNUE_L2_SIZE; ++j)
        out += l2[j] * l3_weights_f_[j];

    return out * NNUE_SCALE / (float)(NNUE_QA * NNUE_QB);
}

bool NNUE::trainInit() {
    // Allocate float copies and scratch buffers
    auto alloc = [&]() -> bool {
        ft_weights_f_ = alignedAlloc<float>(NNUE_FT_OUTPUTS * NNUE_FT_INPUTS);
        ft_biases_f_  = alignedAlloc<float>(NNUE_FT_OUTPUTS);
        l1_weights_f_ = alignedAlloc<float>(NNUE_L1_SIZE * NNUE_FT_OUTPUTS);
        l1_biases_f_  = alignedAlloc<float>(NNUE_L1_SIZE);
        l2_weights_f_ = alignedAlloc<float>(NNUE_L2_SIZE * NNUE_L1_SIZE);
        l2_biases_f_  = alignedAlloc<float>(NNUE_L2_SIZE);
        l3_weights_f_ = alignedAlloc<float>(NNUE_L3_SIZE * NNUE_L2_SIZE);
        l3_biases_f_  = alignedAlloc<float>(NNUE_L3_SIZE);
        scratch_ft_   = alignedAlloc<float>(NNUE_FT_OUTPUTS);
        scratch_l1_   = alignedAlloc<float>(NNUE_L1_SIZE);
        scratch_l2_   = alignedAlloc<float>(NNUE_L2_SIZE);
        return ft_weights_f_ && ft_biases_f_ && l1_weights_f_ && l1_biases_f_
            && l2_weights_f_ && l2_biases_f_ && l3_weights_f_ && l3_biases_f_
            && scratch_ft_ && scratch_l1_ && scratch_l2_;
    };
    if (!alloc()) return false;

    if (weights_loaded_) {
        // Copy from quantized weights: q -> float
        for (size_t i = 0; i < (size_t)NNUE_FT_OUTPUTS * NNUE_FT_INPUTS; ++i)
            ft_weights_f_[i] = (float)ft_weights_[i];
        for (int i = 0; i < NNUE_FT_OUTPUTS; ++i)
            ft_biases_f_[i] = (float)ft_biases_[i];
        for (size_t i = 0; i < (size_t)NNUE_L1_SIZE * NNUE_FT_OUTPUTS; ++i)
            l1_weights_f_[i] = (float)l1_weights_[i];
        for (int i = 0; i < NNUE_L1_SIZE; ++i)
            l1_biases_f_[i] = (float)l1_biases_[i];
        for (size_t i = 0; i < (size_t)NNUE_L2_SIZE * NNUE_L1_SIZE; ++i)
            l2_weights_f_[i] = (float)l2_weights_[i];
        for (int i = 0; i < NNUE_L2_SIZE; ++i)
            l2_biases_f_[i] = (float)l2_biases_[i];
        for (int i = 0; i < NNUE_L2_SIZE; ++i)
            l3_weights_f_[i] = (float)l3_weights_[i];
        l3_biases_f_[0] = (float)l3_biases_[0];
    } else {
        // Xavier/Glorot initialization with positive biases for CRELU
        // Target: hidden layer outputs in the middle of [0, 127] range
        const float ft_scale = std::sqrt(6.0f / (30.0f + NNUE_FT_OUTPUTS));
        const float l1_scale = std::sqrt(6.0f / (NNUE_FT_OUTPUTS + NNUE_L1_SIZE));
        const float l2_scale = std::sqrt(6.0f / (NNUE_L1_SIZE + NNUE_L2_SIZE));
        const float l3_scale = std::sqrt(6.0f / (NNUE_L2_SIZE + 1.0f));
        const float bias_init = NNUE_QB * 64.0f;  // ~4096 -> CRELU midpoint of 64

        uint64_t seed = 42;
        auto rnd = [&]() -> float {
            seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27;
            seed *= 0x2545F4914F6CDD1DULL;
            return (float)((int64_t)(seed % 2001) - 1000) / 1000.0f;
        };
        for (size_t i = 0; i < (size_t)NNUE_FT_OUTPUTS * NNUE_FT_INPUTS; ++i)
            ft_weights_f_[i] = rnd() * ft_scale;
        for (int i = 0; i < NNUE_FT_OUTPUTS; ++i)
            ft_biases_f_[i] = 0.0f;
        for (size_t i = 0; i < (size_t)NNUE_L1_SIZE * NNUE_FT_OUTPUTS; ++i)
            l1_weights_f_[i] = rnd() * l1_scale;
        for (int i = 0; i < NNUE_L1_SIZE; ++i)
            l1_biases_f_[i] = bias_init;
        for (size_t i = 0; i < (size_t)NNUE_L2_SIZE * NNUE_L1_SIZE; ++i)
            l2_weights_f_[i] = rnd() * l2_scale;
        for (int i = 0; i < NNUE_L2_SIZE; ++i)
            l2_biases_f_[i] = bias_init;
        for (int i = 0; i < NNUE_L2_SIZE; ++i)
            l3_weights_f_[i] = rnd() * l3_scale;
        l3_biases_f_[0] = 0.0f;
    }
    return true;
}

void NNUE::trainStep(const uint64_t* pieceBB, float targetScore, float lr) {
    // Extract features
    uint32_t whiteFeat[32], blackFeat[32];
    int wc, bc;
    NNUE::extractFeatures(pieceBB, whiteFeat, wc, blackFeat, bc);

    // Forward pass (float version mirroring the quantized forward)
    // Feature transformer
    std::memcpy(scratch_ft_, ft_biases_f_, NNUE_FT_OUTPUTS * sizeof(float));
    for (int i = 0; i < wc; ++i) {
        const float* col = &ft_weights_f_[whiteFeat[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            scratch_ft_[j] += col[j];
    }

    // Layer 1: 256 -> 32
    for (int i = 0; i < NNUE_L1_SIZE; ++i) {
        float sum = l1_biases_f_[i];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            sum += scratch_ft_[j] * l1_weights_f_[i * NNUE_FT_OUTPUTS + j];
        scratch_l1_[i] = std::max(0.0f, std::min(127.0f, sum / NNUE_QB));
    }

    // Layer 2: 32 -> 32
    for (int i = 0; i < NNUE_L2_SIZE; ++i) {
        float sum = l2_biases_f_[i];
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            sum += scratch_l1_[j] * l2_weights_f_[i * NNUE_L1_SIZE + j];
        scratch_l2_[i] = std::max(0.0f, std::min(127.0f, sum / NNUE_QB));
    }

    // Output
    float out = l3_biases_f_[0];
    for (int j = 0; j < NNUE_L2_SIZE; ++j)
        out += scratch_l2_[j] * l3_weights_f_[j];

    float predScore = out * NNUE_SCALE / (float)(NNUE_QA * NNUE_QB);

    // MSE gradient
    float d_output = 2.0f * (predScore - targetScore);
    float d_score = d_output * NNUE_SCALE / (float)(NNUE_QA * NNUE_QB);

    // Clamp gradient to prevent explosion
    const float max_grad = 100.0f;
    d_score = std::max(-max_grad, std::min(max_grad, d_score));

    // --- Backward pass ---

    // Output layer gradients
    float d_l3_weights[NNUE_L2_SIZE];
    for (int j = 0; j < NNUE_L2_SIZE; ++j)
        d_l3_weights[j] = d_score * scratch_l2_[j];
    float d_l3_bias = d_score;

    // Layer 2 backward (before CRELU)
    float d_l2_raw[NNUE_L2_SIZE];
    for (int j = 0; j < NNUE_L2_SIZE; ++j) {
        d_l2_raw[j] = d_score * l3_weights_f_[j] / NNUE_QB;
    }

    // Apply CRELU backward
    float d_l2[NNUE_L2_SIZE];
    float d_l2_bias[NNUE_L2_SIZE];
    for (int j = 0; j < NNUE_L2_SIZE; ++j) {
        float g = (scratch_l2_[j] > 0.0f && scratch_l2_[j] < 127.0f) ? d_l2_raw[j] : 0.0f;
        d_l2[j] = g;
        d_l2_bias[j] = g;
    }

    // Layer 2 weights gradient
    float d_l2_weights[NNUE_L2_SIZE * NNUE_L1_SIZE];
    for (int i = 0; i < NNUE_L2_SIZE; ++i)
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            d_l2_weights[i * NNUE_L1_SIZE + j] = d_l2[i] * scratch_l1_[j];

    // Layer 1 backward
    float d_l1_raw[NNUE_L1_SIZE];
    std::memset(d_l1_raw, 0, NNUE_L1_SIZE * sizeof(float));
    for (int i = 0; i < NNUE_L2_SIZE; ++i)
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            d_l1_raw[j] += d_l2[i] * l2_weights_f_[i * NNUE_L1_SIZE + j];

    // Apply CRELU backward and scale
    float d_l1[NNUE_L1_SIZE];
    float d_l1_bias[NNUE_L1_SIZE];
    for (int j = 0; j < NNUE_L1_SIZE; ++j) {
        float g = d_l1_raw[j] / NNUE_QB;
        g = (scratch_l1_[j] > 0.0f && scratch_l1_[j] < 127.0f) ? g : 0.0f;
        d_l1[j] = g;
        d_l1_bias[j] = g;
    }

    // Feature transformer gradients
    float d_ft[NNUE_FT_OUTPUTS];
    std::memset(d_ft, 0, NNUE_FT_OUTPUTS * sizeof(float));
    for (int i = 0; i < NNUE_L1_SIZE; ++i)
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            d_ft[j] += d_l1[i] * l1_weights_f_[i * NNUE_FT_OUTPUTS + j];

    // Per-layer learning rates (gradient magnitudes vary vastly across layers)
    // l3 gradients: ~O(1000), l2: ~O(10), l1: ~O(0.001), ft: ~O(0.001)
    float lr_ft = lr * 0.1f;
    float lr_l1 = lr * 0.1f;
    float lr_l2 = lr * 0.0001f;
    float lr_l3 = lr * 0.000001f;

    // --- SGD update ---
    for (int i = 0; i < wc; ++i) {
        float* col = &ft_weights_f_[whiteFeat[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            col[j] -= lr_ft * d_ft[j];
    }
    for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
        ft_biases_f_[j] -= lr_ft * d_ft[j];

    for (int i = 0; i < NNUE_L1_SIZE; ++i) {
        l1_biases_f_[i] -= lr_l1 * d_l1_bias[i];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            l1_weights_f_[i * NNUE_FT_OUTPUTS + j] -= lr_l1 * d_l1[i] * scratch_ft_[j];
    }

    for (int i = 0; i < NNUE_L2_SIZE; ++i) {
        l2_biases_f_[i] -= lr_l2 * d_l2_bias[i];
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            l2_weights_f_[i * NNUE_L1_SIZE + j] -= lr_l2 * d_l2_weights[i * NNUE_L1_SIZE + j];
    }

    for (int j = 0; j < NNUE_L2_SIZE; ++j)
        l3_weights_f_[j] -= lr_l3 * d_l3_weights[j];
    l3_biases_f_[0] -= lr_l3 * d_l3_bias;
}

bool NNUE::saveWeights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    file.write("NNUE", 4);
    int32_t ver = 1;
    file.write(reinterpret_cast<const char*>(&ver), 4);

    // Quantize float weights back to int16/int8 and write
    auto write = [&](const void* buf, size_t sz) {
        file.write(static_cast<const char*>(buf), sz);
    };

    // Feature transformer weights: float -> int16 (clamp to [-32768, 32767])
    auto q16 = [](float v) -> int16_t {
        return (int16_t)std::max(-32768.0f, std::min(32767.0f, std::round(v)));
    };
    auto q8 = [](float v) -> int8_t {
        return (int8_t)std::max(-128.0f, std::min(127.0f, std::round(v)));
    };
    auto q32 = [](float v) -> int32_t {
        return (int32_t)std::max(-2147483648.0f, std::min(2147483647.0f, std::round(v)));
    };

    // Allocate quantized buffers
    int16_t* q_ft_w = alignedAlloc<int16_t>(NNUE_FT_OUTPUTS * NNUE_FT_INPUTS);
    int16_t* q_ft_b = alignedAlloc<int16_t>(NNUE_FT_OUTPUTS);
    int8_t*  q_l1_w = alignedAlloc<int8_t>(NNUE_L1_SIZE * NNUE_FT_OUTPUTS);
    int32_t* q_l1_b = alignedAlloc<int32_t>(NNUE_L1_SIZE);
    int8_t*  q_l2_w = alignedAlloc<int8_t>(NNUE_L2_SIZE * NNUE_L1_SIZE);
    int32_t* q_l2_b = alignedAlloc<int32_t>(NNUE_L2_SIZE);
    int8_t*  q_l3_w = alignedAlloc<int8_t>(NNUE_L2_SIZE);
    int32_t* q_l3_b = alignedAlloc<int32_t>(1);

    if (!q_ft_w || !q_ft_b || !q_l1_w || !q_l1_b || !q_l2_w || !q_l2_b || !q_l3_w || !q_l3_b) {
        alignedFree(q_ft_w); alignedFree(q_ft_b); alignedFree(q_l1_w);
        alignedFree(q_l1_b); alignedFree(q_l2_w); alignedFree(q_l2_b);
        alignedFree(q_l3_w); alignedFree(q_l3_b);
        return false;
    }

    for (size_t i = 0; i < (size_t)NNUE_FT_OUTPUTS * NNUE_FT_INPUTS; ++i)
        q_ft_w[i] = q16(ft_weights_f_[i]);
    for (int i = 0; i < NNUE_FT_OUTPUTS; ++i)
        q_ft_b[i] = q16(ft_biases_f_[i]);
    for (size_t i = 0; i < (size_t)NNUE_L1_SIZE * NNUE_FT_OUTPUTS; ++i)
        q_l1_w[i] = q8(l1_weights_f_[i]);
    for (int i = 0; i < NNUE_L1_SIZE; ++i)
        q_l1_b[i] = q32(l1_biases_f_[i]);
    for (size_t i = 0; i < (size_t)NNUE_L2_SIZE * NNUE_L1_SIZE; ++i)
        q_l2_w[i] = q8(l2_weights_f_[i]);
    for (int i = 0; i < NNUE_L2_SIZE; ++i)
        q_l2_b[i] = q32(l2_biases_f_[i]);
    for (int i = 0; i < NNUE_L2_SIZE; ++i)
        q_l3_w[i] = q8(l3_weights_f_[i]);
    q_l3_b[0] = q32(l3_biases_f_[0]);

    write(q_ft_w, NNUE_FT_OUTPUTS * NNUE_FT_INPUTS * sizeof(int16_t));
    write(q_ft_b, NNUE_FT_OUTPUTS * sizeof(int16_t));
    write(q_l1_w, NNUE_L1_SIZE * NNUE_FT_OUTPUTS * sizeof(int8_t));
    write(q_l1_b, NNUE_L1_SIZE * sizeof(int32_t));
    write(q_l2_w, NNUE_L2_SIZE * NNUE_L1_SIZE * sizeof(int8_t));
    write(q_l2_b, NNUE_L2_SIZE * sizeof(int32_t));
    write(q_l3_w, NNUE_L3_SIZE * NNUE_L2_SIZE * sizeof(int8_t));
    write(q_l3_b, NNUE_L3_SIZE * sizeof(int32_t));

    alignedFree(q_ft_w); alignedFree(q_ft_b); alignedFree(q_l1_w);
    alignedFree(q_l1_b); alignedFree(q_l2_w); alignedFree(q_l2_b);
    alignedFree(q_l3_w); alignedFree(q_l3_b);

    return true;
}
