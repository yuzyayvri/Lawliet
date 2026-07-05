#pragma once
#include "../core/board.hpp"
#include "../ai/time_manager.hpp"
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>

#pragma pack(push, 1)
// Polyglot Opening Book Entry Layout
struct BookEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};
#pragma pack(pop)

// ============================================================================
// OPENING BOOK LOADER & ENGINE LOOKUP
// ============================================================================
class OpeningBook {
private:
    std::vector<BookEntry> entries;

    static inline uint16_t swap16(uint16_t val) { return __builtin_bswap16(val); }
    static inline uint64_t swap64(uint64_t val) { return __builtin_bswap64(val); }

public:
    bool load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            std::vector<std::string> fallbacks = {
                "book.bin", "../book.bin", "../../book.bin",
                "bin/book.bin", "../bin/book.bin", "../../bin/book.bin",
                "gm2001.bin", "../gm2001.bin", "../../gm2001.bin"
            };
            for (const auto& path : fallbacks) {
                if (path == filename) continue;
                file.clear();
                file.open(path, std::ios::binary | std::ios::ate);
                if (file) break;
            }
        }
        if (!file) {
            std::cout << "info string Warning: Opening book file not found in any standard fallback paths." << std::endl;
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t entryCount = size / sizeof(BookEntry);
        entries.resize(entryCount);

        if (!file.read(reinterpret_cast<char*>(entries.data()), size)) {
            entries.clear();
            std::cout << "info string Error: Failed to read opening book entries." << std::endl;
            return false;
        }

        for (auto& entry : entries) {
            entry.key = swap64(entry.key);
            entry.move = swap16(entry.move);
            entry.weight = swap16(entry.weight);
        }

        std::sort(entries.begin(), entries.end(), [](const BookEntry& a, const BookEntry& b) {
            return a.key < b.key;
        });

        std::cout << "info string Opening book loaded successfully with " << entryCount << " entries." << std::endl;
        return true;
    }

    bool isOpen() const { return !entries.empty(); }

    uint64_t computePolyglotHash(const Board& board) const;
    uint16_t lookup(uint64_t polyKey) const;
};

// ============================================================================
// SEARCH CONTEXT & MOVE ORDERING HEURISTICS
// ============================================================================
struct SearchContext {
    // Killer Moves Heuristic: prioritizes quiet moves that caused cutoffs in sibling branches
    Move killerMoves[128][2]{};

    // Piece-To-History Heuristic: history scores indexed by [color][piece_type][to_square]
    int historyTable[2][6][64]{};

    // Countermove Heuristic: quiet moves stored to refute the opponent's previous move
    Move counterMoveTable[64][64]{};

    // Continuation History Heuristic: quiet history based on the previous piece's movement
    int continuationHistory[12][64]{};

    // Static Evaluation Correction History (CorrHist): learns and refines evaluation biases
    int corrHist[2][16384]{};

    // Bounded Repetition Detection Hash Stack
    uint64_t hashStack[4096]{};
    int hashStackIdx = 0;

    // Search buffers to minimize heap allocations per thread
    Move moveBuffers[210][256]{};
    int moveCounts[210]{};

    int64_t localNodes = 0;
    Move rootBestMove{};

    int64_t ttLookups = 0;
    int64_t ttHits = 0;
    int64_t fhf = 0;
    int64_t failHighs = 0;
    int maxQuiescencePly = 0;
    int bestScore = 0;
    int threadId = 0; // Diversifies Lazy SMP search branches
};

// ============================================================================
// PAWN HASH TABLE (PHT) STRUCTURE
// ============================================================================
struct PawnEntry {
    uint64_t key = 0;          // Unique pawn skeleton Zobrist key
    int mgScore = 0;           // Cached static pawn score (Middlegame)
    int egScore = 0;           // Cached static pawn score (Endgame)
    uint64_t wPassedPawns = 0; // Bitboard of White passed pawns
    uint64_t bPassedPawns = 0; // Bitboard of Black passed pawns
};

// ============================================================================
// LAWLIET CHESS ENGINE CORE INTERFACE
// ============================================================================
class Lawliet {
private:
    int maxDepth;
    OpeningBook book;
    TimeManager* activeTm = nullptr;
    static constexpr int INF = 10000000;
    static constexpr int MAX_PLY = 128;
    static constexpr int MAX_QDEPTH = 64;
    static constexpr int MAX_TOTAL_PLY = MAX_PLY + MAX_QDEPTH + 10;

    // Transposition Table: Shared, lock-free generational cache (~67MB footprint)
    static constexpr int TT_SIZE = 1 << 23;
    static constexpr int PAWN_SIZE = 16384; // Compact and L1/L2 cache-friendly size

    enum TTFlag : uint8_t { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };
    struct TTEntry {
        std::atomic<uint64_t> data{0}; // Key signature and move data merged into a single word

        TTEntry() : data(0) {}
        TTEntry(const TTEntry& other) {
            data.store(other.data.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        TTEntry& operator=(const TTEntry& other) {
            data.store(other.data.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

    uint8_t ttAge = 0; // Age key for deep/generational replacement strategy

    static uint64_t fileMasks[8];
    static uint64_t rankMasks[8];
    static uint64_t pawnPassedMask[2][64];
    static uint64_t pawnIsolatedMask[64];

    static int mvvLva[6][7];
    static bool tablesInitialized;
    static int lmrTable[128][256];

    std::vector<TTEntry> transpositionTable;
    mutable std::vector<PawnEntry> pawnTable; // Lock-free mutable PHT

    uint64_t zobristPiece[64][12]{};
    uint64_t zobristCastle[16]{};
    uint64_t zobristEp[64]{};
    uint64_t zobristSide{};

    Move pvMove{};

    static void initTables();
    static inline int pieceToZobristIndex(int piece) {
        if (piece == 0) return -1;
        int type = std::abs(piece) - 1;
        return piece > 0 ? type : type + 6;
    }
    static int encodeCastling(const Board& board);
    static int scoreToTT(int score, int ply);
    static int scoreFromTT(int score, int ply);

    // Packs search depth, flag, move squares, and age into a single 64-bit value
    static inline uint64_t packData(int score, int depth, uint8_t flag, uint16_t fromSq, uint16_t toSq, int16_t promo, uint8_t age, uint16_t keySig) {
        uint64_t s = static_cast<uint16_t>(score); // Bits 0-15
        uint64_t d = static_cast<uint64_t>(std::clamp(depth, 0, 255)) << 16; // Bits 16-23
        uint64_t flg = static_cast<uint64_t>(flag & 3) << 24; // Bits 24-25
        uint64_t fSq = static_cast<uint64_t>(fromSq & 0x3F) << 26; // Bits 26-31
        uint64_t tSq = static_cast<uint64_t>(toSq & 0x3F) << 32; // Bits 32-37
        uint64_t p = static_cast<uint64_t>((promo + 8) & 0xF) << 38; // Bits 38-41
        uint64_t a = static_cast<uint64_t>(age & 0x3F) << 42; // Bits 42-47
        uint64_t kSig = static_cast<uint64_t>(keySig) << 48; // Bits 48-63
        return s | d | flg | fSq | tSq | p | a | kSig;
    }

    static inline void unpackData(uint64_t val, int& score, int& depth, uint8_t& flag, uint16_t& fromSq, uint16_t& toSq, int16_t& promo, uint8_t& age, uint16_t& keySig) {
        score = static_cast<int16_t>(val & 0xFFFFULL);
        depth = static_cast<int>((val >> 16) & 0xFF);
        flag = static_cast<uint8_t>((val >> 24) & 3);
        fromSq = static_cast<uint16_t>((val >> 26) & 0x3F);
        toSq = static_cast<uint16_t>((val >> 32) & 0x3F);
        promo = static_cast<int16_t>(((val >> 38) & 0xF) - 8);
        age = static_cast<uint8_t>((val >> 42) & 0x3F);
        keySig = static_cast<uint16_t>((val >> 48) & 0xFFFFULL);
    }

    // Static Exchange Evaluation (SEE) Algorithm
    int see(const Board& board, int from, int to) const;
    uint64_t seeAttackers(const Board& board, int sq, uint64_t occupied) const;
    uint64_t seeXrays(const Board& board, int sq, uint64_t occupied) const;
    uint64_t getPinnedPieces(const Board& board, int kingSq, int friendlyColor) const;

    uint64_t computeHash(const Board& board) const;
    int evaluateBoard(const Board& board, int alpha = -INF, int beta = INF, const SearchContext* ctx = nullptr) const;
    void generateLegalMoves(const Board& board, int color, Move* out, int& count) const;
    void generateCaptures(const Board& board, int color, Move* out, int& count) const;
    void doMove(Board& board, Move& m, uint64_t& hash, SearchContext& ctx);
    void undoMove(Board& board, Move& m, uint64_t& hash, SearchContext& ctx);
    void storeTT(uint64_t key, int depth, int score, TTFlag flag, const Move& bestMove, int ply);
    bool probeTT(uint64_t key, int depth, int alpha, int beta, int& scoreOut, Move& bestMoveOut, int ply, SearchContext& ctx);
    int scoreMove(const Move& m, const Board& board, int ply, const Move& ttMove, const SearchContext& ctx) const;
    void orderMoves(Move* moves, int count, const Board& board, int ply, const Move& ttMove, const SearchContext& ctx) const;

    std::string extractPv(Board& board, uint64_t hash);
    std::string squareToUci(int sq);

    // Alpha-Beta Search Phases
    int quiescence(Board& board, int alpha, int beta, int ply, uint64_t hash, TimeManager& tm, SearchContext& ctx);
    int negamax(Board& board, int depth, int alpha, int beta, int ply, uint64_t hash, TimeManager& tm, SearchContext& ctx, Move excludedMove = Move{});

    // Lazy SMP Multithreading Handlers
    Move thinkThread(Board& board, TimeManager& tm, SearchContext& ctx, int threadId);
    void searchWorker(Board board, TimeManager& tm, int threadId, Move& outBestMove);

public:
    explicit Lawliet(int depth = 64);
    void setDepth(int depth);
    void reset();
    void loadBook(const std::string& path);

    Move think(Board& board);
    Move think(Board& board, TimeManager& tm);
};
