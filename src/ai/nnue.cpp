#include "nnue.hpp"
#include <cstdlib>
#include <cstring>

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
