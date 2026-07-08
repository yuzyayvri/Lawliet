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
                "gm2001.bin", "../gm2001.bin", "../../gm2001.bin",
                "../src/book.bin", "../..src/book.bin"
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

    // Static Evaluation Correction History (CorrHist)
    int corrHist[2][16384]{};

    // Bounded Repetition Detection Hash Stack
    uint64_t hashStack[4096]{};
    int hashStackIdx = 0;
    int rootLastIrreversible = 0;

    // Thread-local Pawn Hash Table (PHT) to ensure thread-safety
    mutable PawnEntry pawnTable[16384]{};

    // Search buffers to minimize heap allocations per thread
    Move moveBuffers[210][256]{};
    int moveScores[210][256]{}; // Parallel scores array
    int moveCounts[210]{};

    int64_t localNodes = 0;
    Move rootBestMove{};

    // Core search statistics
    int64_t ttLookups = 0;
    int64_t ttHits = 0;
    int64_t ttExactHits = 0;
    int64_t ttLowerBoundHits = 0;
    int64_t ttUpperBoundHits = 0;
    int64_t ttCutoffs = 0;
    int64_t ttMoveUsed = 0;
    int64_t ttMoveBetaCutoffs = 0;
    int64_t ttStores = 0;
    int64_t ttReplacements = 0;
    int64_t ttAgeCollisions = 0;
    int64_t ttOccupancy = 0;
    int64_t ttMaxAge = 0;
    int64_t nodesSearched = 0;
    int64_t quiescenceNodes = 0;
    int64_t failHighs = 0;
    int64_t failLows = 0;
    int64_t exactNodes = 0;
    int64_t betaCutoffs = 0;
    int64_t alphaFailures = 0;
    int64_t cutoffDepthSum = 0;
    int64_t fhf = 0;

    // Static evaluation pipeline statistics
    int64_t staticEvalCalls = 0;
    int64_t incrementalUpdates = 0;
    int64_t fullRecomputations = 0;
    int64_t staticEvalSum = 0;
    int staticEvalMax = -9999999;
    int staticEvalMin = 9999999;

    // Principal variation statistics
    int64_t pvChanges = 0;
    int64_t rootBestMoveChanges = 0;
    int64_t pvLengthSum = 0;

    // Move ordering placement tracking
    int64_t bestMoveRankTotal = 0;
    int64_t bestMoveRankCount = 0;
    int64_t moveOrderCounts[7]{};
    int64_t ttMoveUsedCount = 0;
    int64_t winningCaptureCount = 0;
    int64_t killerMoveCount = 0;
    int64_t historyHeuristicCount = 0;
    int64_t countermoveCount = 0;
    int64_t otherOrderCount = 0;
    struct MoveOrderingStats {
        int64_t bestMoveRankTotal = 0;
        int64_t bestMoveRankCount = 0;
        int64_t moveOrderCounts[7] = {};
        int64_t firstMoveCutoffs = 0;
        int avgBestMoveRank = 0;
        int medianBestMoveRank = 0;
    } moveOrderStats;

    // Aspiration window statistics
    int64_t aspirationFailHighs = 0;
    int64_t aspirationFailLows = 0;
    int64_t aspirationWindowSum = 0;
    int64_t fullWindowSearches = 0;
    int64_t nullWindowSearches = 0;

    // History heuristic statistics
    int64_t historyHits = 0;
    int64_t historyCutoffs = 0;
    int64_t historyScoreSum = 0;

    // Killer heuristic statistics
    int64_t killerHits = 0;
    int64_t killerCutoffs = 0;
    int64_t killerRankSum = 0;

    // Pruning statistics
    int64_t nullMoveAttempts = 0;
    int64_t nullMoveSuccess = 0;
    int64_t lmrApplications = 0;
    int64_t lmrReductionsSum = 0;
    int64_t lmrMaxReduction = 0;
    int64_t lmrReducedMoves = 0;
    int64_t lmrSuccessfulReSearches = 0;
    int64_t lmrApplicationsCount = 0;
    int64_t futilityApplications = 0;
    int64_t reverseFutilityApplications = 0;
    int64_t razoringApplications = 0;
    int64_t probCutAttempts = 0;
    int64_t probCutSuccess = 0;
    int64_t singularExtensionAttempts = 0;
    int64_t singularExtensionSuccess = 0;

    // Extension statistics
    int64_t checkExtensions = 0;
    int64_t recaptureExtensions = 0;
    int64_t passedPawnExtensions = 0;
    int64_t singularExtensions = 0;
    int64_t extensionSizeSum = 0;

    // SEE statistics
    int64_t seeCalls = 0;
    int64_t seeAccepted = 0;
    int64_t seeRejected = 0;

    // Time management statistics
    int64_t allocatedTime = 0;
    int64_t actualTime = 0;
    int64_t softStops = 0;
    int64_t hardStops = 0;
    int64_t timeExpiredDuringSearch = 0;

    // Lazy SMP statistics
    int64_t helperThreadUtil = 0;
    int64_t splitPoints = 0;
    int64_t ttSharingRate = 0;
    int64_t threadIdleTime = 0;
    int64_t threadStopEvents = 0;

    int maxQuiescencePly = 0;
    int bestScore = 0;
    int threadId = 0; // Diversifies Lazy SMP search branches

    // Root iteration recording
    struct RootIterationInfo {
        Move move;
        int initialOrder = 0;
        int finalOrder = 0;
        int finalScore = 0;
        int64_t nodes = 0;
        int time = 0;
        int depth = 0;
        int reSearches = 0;
        int failHighCount = 0;
    };
    std::vector<RootIterationInfo> rootIterations;
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

    // Transposition Table: Shared, lock-free generational cache (~128MB footprint)
    static constexpr int TT_SIZE = 1 << 23;
    static constexpr int PAWN_SIZE = 16384; // Compact and L1/L2 cache-friendly size

    enum TTFlag : uint8_t { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };
    struct TTEntry {
        std::atomic<uint64_t> key{0};
        std::atomic<uint64_t> data{0};

        TTEntry() : key(0), data(0) {}
        TTEntry(const TTEntry& other) {
            key.store(other.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
            data.store(other.data.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        TTEntry& operator=(const TTEntry& other) {
            key.store(other.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
            data.store(other.data.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

    std::atomic<uint8_t> ttAge{0}; // Age key for deep/generational replacement strategy

    static uint64_t fileMasks[8];
    static uint64_t rankMasks[8];
    static uint64_t pawnPassedMask[2][64];
    static uint64_t pawnIsolatedMask[64];

    static int mvvLva[6][7];
    static bool tablesInitialized;
    static int lmrTable[128][256];

    std::vector<TTEntry> transpositionTable;

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
    static inline uint64_t packData(int score, int depth, uint8_t flag, uint16_t fromSq, uint16_t toSq, int16_t promo, uint8_t age) {
        uint64_t s = static_cast<uint32_t>(score); // Bits 0-31
        uint64_t d = static_cast<uint64_t>(std::clamp(depth, 0, 255)) << 32; // Bits 32-39
        uint64_t flg = static_cast<uint64_t>(flag & 3) << 40; // Bits 40-41
        uint64_t fSq = static_cast<uint64_t>(fromSq & 0x3F) << 42; // Bits 42-47
        uint64_t tSq = static_cast<uint64_t>(toSq & 0x3F) << 48; // Bits 48-53
        uint64_t p = static_cast<uint64_t>((promo + 8) & 0xF) << 54; // Bits 54-57
        uint64_t a = static_cast<uint64_t>(age & 0x3F) << 58; // Bits 58-63
        return s | d | flg | fSq | tSq | p | a;
    }

    static inline void unpackData(uint64_t val, int& score, int& depth, uint8_t& flag, uint16_t& fromSq, uint16_t& toSq, int16_t& promo, uint8_t& age) {
        score = static_cast<int>(static_cast<int32_t>(val & 0xFFFFFFFFULL));
        depth = static_cast<int>((val >> 32) & 0xFF);
        flag = static_cast<uint8_t>((val >> 40) & 3);
        fromSq = static_cast<uint16_t>((val >> 42) & 0x3F);
        toSq = static_cast<uint16_t>((val >> 48) & 0x3F);
        promo = static_cast<int16_t>(((val >> 54) & 0xF) - 8);
        age = static_cast<uint8_t>((val >> 58) & 0x3F);
    }

    // Static Exchange Evaluation (SEE) Algorithm
    int see(const Board& board, int from, int to, SearchContext& ctx) const;
    uint64_t seeAttackers(const Board& board, int sq, uint64_t occupied) const;
    uint64_t seeXrays(const Board& board, int sq, uint64_t occupied) const;
    uint64_t getPinnedPieces(const Board& board, int kingSq, int friendlyColor) const;

    uint64_t computeHash(const Board& board) const;
    void generateLegalMoves(const Board& board, int color, Move* out, int& count) const;
    void generateCaptures(const Board& board, int color, Move* out, int& count) const;
    void doMove(Board& board, Move& m, uint64_t& hash, SearchContext& ctx);
    void undoMove(Board& board, Move& m, uint64_t& hash, SearchContext& ctx);
    void storeTT(uint64_t key, int depth, int score, TTFlag flag, const Move& bestMove, int ply, SearchContext& ctx);
    bool probeTT(uint64_t key, int depth, int alpha, int beta, int& scoreOut, Move& bestMoveOut, int ply, SearchContext& ctx);
    int scoreMove(const Move& m, const Board& board, int ply, const Move& ttMove, SearchContext& ctx) const;
    void orderMoves(Move* moves, int* scores, int count, const Board& board, int ply, const Move& ttMove, SearchContext& ctx) const;

    std::string extractPv(Board& board, uint64_t hash);
    // Alpha-Beta Search Phases
    int quiescence(Board& board, int alpha, int beta, int ply, uint64_t hash, TimeManager& tm, SearchContext& ctx);
    int negamax(Board& board, int depth, int alpha, int beta, int ply, uint64_t hash, TimeManager& tm, SearchContext& ctx, int lastIrreversible, Move excludedMove = Move{});

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
    int evaluateBoard(const Board& board, int alpha = -INF, int beta = INF, const SearchContext* ctx = nullptr) const;
    static std::string squareToUci(int sq);
};
