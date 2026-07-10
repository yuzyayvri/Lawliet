#include "nnue.hpp"
#include <cstdlib>
#include <cstring>

NNUE::NNUE() {}
NNUE::~NNUE() {
    alignedFreeNNUE(ft_weights_); alignedFreeNNUE(ft_biases_);
    alignedFreeNNUE(l1_weights_); alignedFreeNNUE(l1_biases_);
    alignedFreeNNUE(l2_weights_); alignedFreeNNUE(l2_biases_);
    alignedFreeNNUE(l3_weights_); alignedFreeNNUE(l3_biases_);
}

bool NNUE::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    // --- Stockfish NNUE binary format (v2) header ---
    // Bytes 0-7:   64-bit network hash identifier
    // Bytes 8-11:  uint32 LE description length
    // Bytes 12+:   description string (not null-terminated)
    // Then:        raw weight data in order: ft_w, ft_b, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b

    char hash[8];
    file.read(hash, 8);  // skip hash

    int32_t descLen;
    file.read(reinterpret_cast<char*>(&descLen), 4);
    if (descLen <= 0 || descLen > 1024) return false;

    // Validate the description string contains expected architecture
    std::string desc(descLen, '\0');
    file.read(desc.data(), descLen);
    if (desc.find("Features=HalfKP") == std::string::npos) {
        std::cerr << "info string NNUE error: unexpected network architecture: "
                  << desc.substr(0, 80) << std::endl;
        return false;
    }

    // --- Allocate aligned buffers ---
    auto alloc = [&]() -> bool {
        ft_weights_ = alignedAllocNNUE<int16_t>(NNUE_FT_INPUTS * (size_t)NNUE_FT_OUTPUTS);
        ft_biases_  = alignedAllocNNUE<int16_t>(NNUE_FT_OUTPUTS);
        l1_weights_ = alignedAllocNNUE<int8_t>(NNUE_L1_SIZE * (size_t)NNUE_FT_TOTAL);
        l1_biases_  = alignedAllocNNUE<int32_t>(NNUE_L1_SIZE);
        l2_weights_ = alignedAllocNNUE<int8_t>(NNUE_L2_SIZE * (size_t)NNUE_L1_SIZE);
        l2_biases_  = alignedAllocNNUE<int32_t>(NNUE_L2_SIZE);
        l3_weights_ = alignedAllocNNUE<int8_t>(NNUE_L3_SIZE * (size_t)NNUE_L2_SIZE);
        l3_biases_  = alignedAllocNNUE<int32_t>(NNUE_L3_SIZE);
        return ft_weights_ && ft_biases_ && l1_weights_ && l1_biases_
            && l2_weights_ && l2_biases_ && l3_weights_ && l3_biases_;
    };
    if (!alloc()) return false;

    // --- Read layers ---
    auto read = [&](void* buf, size_t sz) {
        file.read(static_cast<char*>(buf), sz);
    };

    // Stockfish 13 file format: all layers store biases BEFORE weights
    read(ft_biases_,  NNUE_FT_OUTPUTS * sizeof(int16_t));
    read(ft_weights_, NNUE_FT_INPUTS * (size_t)NNUE_FT_OUTPUTS * sizeof(int16_t));
    read(l1_biases_,  NNUE_L1_SIZE * sizeof(int32_t));
    read(l1_weights_, (size_t)NNUE_L1_SIZE * NNUE_FT_TOTAL * sizeof(int8_t));
    read(l2_biases_,  NNUE_L2_SIZE * sizeof(int32_t));
    read(l2_weights_, (size_t)NNUE_L2_SIZE * NNUE_L1_SIZE * sizeof(int8_t));
    read(l3_biases_,  NNUE_L3_SIZE * sizeof(int32_t));
    read(l3_weights_, (size_t)NNUE_L3_SIZE * NNUE_L2_SIZE * sizeof(int8_t));

    weights_loaded_ = true;
    return true;
}

// Stockfish 13 HalfKP(Friend) feature extraction
// pieceBB layout: 0=P_w,1=N_w,2=B_w,3=R_w,4=Q_w,5=K_w,6=P_b,7=N_b,8=B_b,9=R_b,10=Q_b,11=K_b
//
// HalfKP(Friend) feature index formula (from Stockfish 13 source):
//   index = orient(perspective, pieceSq) + kpp_board_index[perspective][piece]
//           + PS_END * orient(perspective, kingSq)
// where:
//   - orient(perspective, sq) = sq ^ (perspective * 63)  (rotate 180° for BLACK)
//   - PS_END = PS_W_KING = 641
//   - kpp_board_index[WHITE] (perspective=0): W_PAWN=1, W_KNIGHT=129, W_BISHOP=257,
//     W_ROOK=385, W_QUEEN=513, B_PAWN=65, B_KNIGHT=193, B_BISHOP=321, B_ROOK=449,
//     B_QUEEN=577
//   - kpp_board_index[BLACK] (perspective=1): colors reversed (B pieces become "us")
//
// IMPORTANT: The "Friend" in HalfKP(Friend) refers to the KING being the
// friend (side-to-move) king, NOT to piece filtering. ALL non-king pieces
// (both colors) are included for each perspective!
void NNUE::extractFeatures(const uint64_t* pieceBB,
                           uint32_t* whiteFeat, int& wCount,
                           uint32_t* blackFeat, int& bCount) {
    wCount = bCount = 0;

    int wKing = pieceBB[5] ? __builtin_ctzll(pieceBB[5]) : -1;
    int bKing = pieceBB[11] ? __builtin_ctzll(pieceBB[11]) : -1;
    if (wKing < 0 || bKing < 0) return;

    // kpp_board_index for WHITE perspective (perspective=0):
    //   white pieces are "us", black pieces are "them"
    // kpp_board_index for BLACK perspective (perspective=1):
    //   colors reversed: black pieces become "us", white pieces become "them"
    // See Stockfish 13 nnue_common.h for the full table.
    static const int psWhite[12] = {  1, 129, 257, 385, 513, -1,  65, 193, 321, 449, 577, -1};
    static const int psBlack[12] = { 65, 193, 321, 449, 577, -1,   1, 129, 257, 385, 513, -1};

    // King squares oriented to each perspective:
    //   orient(WHITE, sq) = sq  (identity)
    //   orient(BLACK, sq) = sq ^ 63  (rotate 180°)
    int wKsq = wKing;
    int bKsq = bKing ^ 63;

    // White perspective: ALL non-king pieces, identity orientation
    for (int i = 0; i < 12; ++i) {
        if (psWhite[i] < 0) continue;  // skip kings (indices 5, 11)
        uint64_t bb = pieceBB[i];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            whiteFeat[wCount++] = halfKPIndex(wKsq, sq, psWhite[i]);
        }
    }

    // Black perspective: ALL non-king pieces, rotated 180°
    for (int i = 0; i < 12; ++i) {
        if (psBlack[i] < 0) continue;  // skip kings (indices 5, 11)
        uint64_t bb = pieceBB[i];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            blackFeat[bCount++] = halfKPIndex(bKsq, sq ^ 63, psBlack[i]);
        }
    }
}

int NNUE::forward(const uint8_t* values) const {
    // Hidden Layer 1: 512 -> 32 (ClippedReLU: x >> 6, clamp to [0,127])
    int32_t l1[NNUE_L1_SIZE];
    for (int i = 0; i < NNUE_L1_SIZE; ++i) {
        int32_t sum = l1_biases_[i];
        for (int j = 0; j < NNUE_FT_TOTAL; ++j)
            sum += (int32_t)values[j] * (int32_t)l1_weights_[i * NNUE_FT_TOTAL + j];
        l1[i] = creluHidden(sum);
    }

    // Hidden Layer 2: 32 -> 32 (ClippedReLU: x >> 6, clamp to [0,127])
    int32_t l2[NNUE_L2_SIZE];
    for (int i = 0; i < NNUE_L2_SIZE; ++i) {
        int32_t sum = l2_biases_[i];
        for (int j = 0; j < NNUE_L1_SIZE; ++j)
            sum += l1[j] * (int32_t)l2_weights_[i * NNUE_L1_SIZE + j];
        l2[i] = creluHidden(sum);
    }

    // Output: 32 -> 1 (linear, no activation)
    int32_t out = l3_biases_[0];
    for (int j = 0; j < NNUE_L2_SIZE; ++j)
        out += l2[j] * (int32_t)l3_weights_[j];

    // Scale to centipawns (Stockfish 13): raw / FV_SCALE
    return out / NNUE_FV_SCALE;
}

int NNUE::evaluate(const uint32_t* whiteFeat, int wCount,
                   const uint32_t* blackFeat, int bCount,
                   bool whiteToMove) const {
    if (!weights_loaded_) return 0;

    // Build two separate [256] accumulators, one for each king perspective,
    // then concatenate into a single [512] array for the hidden layer.
    int32_t accWhite[NNUE_FT_OUTPUTS];
    int32_t accBlack[NNUE_FT_OUTPUTS];

    // Start from the shared FT biases
    for (int j = 0; j < NNUE_FT_OUTPUTS; ++j) {
        int32_t b = (int32_t)ft_biases_[j];
        accWhite[j] = b;
        accBlack[j] = b;
    }

    // Accumulate white perspective features
    for (int i = 0; i < wCount; ++i) {
        const int16_t* col = &ft_weights_[whiteFeat[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            accWhite[j] += (int32_t)col[j];
    }

    // Accumulate black perspective features
    for (int i = 0; i < bCount; ++i) {
        const int16_t* col = &ft_weights_[blackFeat[i] * (size_t)NNUE_FT_OUTPUTS];
        for (int j = 0; j < NNUE_FT_OUTPUTS; ++j)
            accBlack[j] += (int32_t)col[j];
    }

    // Stockfish 13 orders perspectives: [side_to_move, ~side_to_move]
    // whiteToMove: first 256 = white accumulator, last 256 = black accumulator
    // blackToMove: first 256 = black accumulator, last 256 = white accumulator
    const int32_t* first  = whiteToMove ? accWhite : accBlack;
    const int32_t* second = whiteToMove ? accBlack : accWhite;

    // Clamp to uint8 [0..127] range (Stockfish 13 FT CReLU) and interleave
    uint8_t combined[NNUE_FT_TOTAL];
    for (int j = 0; j < NNUE_FT_OUTPUTS; ++j) {
        combined[j]                      = (uint8_t)creluFT(first[j]);
        combined[j + NNUE_FT_OUTPUTS]    = (uint8_t)creluFT(second[j]);
    }

    return forward(combined);
}
