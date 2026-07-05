#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include "lawliet.hpp"
#include "time_manager.hpp"
#include <cstring>
#include <random>
#include <algorithm>
#include <memory>
#include <cmath>
#include <iomanip>

static uint64_t polyglotPiece[12][64];
static uint64_t polyglotCastle[4];
static uint64_t polyglotEp[8];
static uint64_t polyglotSide;
uint64_t Lawliet::fileMasks[8];
uint64_t Lawliet::rankMasks[8];
uint64_t Lawliet::pawnPassedMask[2][64];
uint64_t Lawliet::pawnIsolatedMask[64];
int Lawliet::lmrTable[128][256];
static bool polyglotKeysInitialized = false;
static uint32_t poly_seed = 15071982;

static uint32_t poly_random_32() {
    poly_seed = poly_seed * 1103515245 + 12345;
    return (poly_seed >> 16) & 0x7FFF;
}
static uint64_t poly_random_64() {
    uint64_t r = 0;
    r |= (uint64_t)poly_random_32() << 0;
    r |= (uint64_t)poly_random_32() << 15;
    r |= (uint64_t)poly_random_32() << 30;
    r |= (uint64_t)poly_random_32() << 45;
    r |= ((uint64_t)poly_random_32() & 0xF) << 60;
    return r;
}
static void initPolyglotKeys() {
    if (polyglotKeysInitialized) return;
    poly_seed = 15071982;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 64; ++j) polyglotPiece[i][j] = poly_random_64();
    for (int i = 0; i < 4; ++i) polyglotCastle[i] = poly_random_64();
    for (int i = 0; i < 8; ++i) polyglotEp[i] = poly_random_64();
    polyglotSide = poly_random_64();
    polyglotKeysInitialized = true;
}

static void printSearchStats(const SearchContext& ctx, int depth, int score, int elapsedMs, int64_t totalNodes) {
    double nps = elapsedMs > 0 ? (static_cast<double>(totalNodes) * 1000.0 / elapsedMs) : 0.0;
    double ttHitRate = ctx.ttLookups > 0 ? (static_cast<double>(ctx.ttHits) * 100.0 / ctx.ttLookups) : 0.0;
    double orderQuality = ctx.failHighs > 0 ? (static_cast<double>(ctx.fhf) * 100.0 / ctx.failHighs) : 0.0;

    std::cout << "info string ================ Lawliet Search Stats ================" << std::endl;
    std::cout << "info string   Depth:         " << depth << " / Seldepth: " << ctx.maxQuiescencePly << std::endl;
    if (std::abs(score) > 9000000) {
        int mateIn = (score > 0) ? (10000000 - score + 1) / 2 : -(10000000 + score + 1) / 2;
        std::cout << "info string   Score:         Mate in " << mateIn << std::endl;
    } else {
        std::cout << "info string   Score:         " << score << " cp" << std::endl;
    }
    std::cout << "info string   Nodes Searched: " << totalNodes << std::endl;
    std::cout << "info string   NPS:           " << static_cast<int64_t>(nps) << std::endl;
    std::cout << "info string   TT Lookups:    " << ctx.ttLookups << " | TT Hits: " << ctx.ttHits << " (" << std::fixed << std::setprecision(1) << ttHitRate << "%)" << std::endl;
    std::cout << "info string   Move Ordering: First-Move-Cutoffs (FHF): " << ctx.fhf << " / " << ctx.failHighs << " (" << std::fixed << std::setprecision(1) << orderQuality << "%)" << std::endl;
    std::cout << "info string ======================================================" << std::endl;
}

int Lawliet::mvvLva[6][7];
bool Lawliet::tablesInitialized = false;

static inline int kingDistance(int sq1, int sq2) {
    return std::max(std::abs(sq1 / 8 - sq2 / 8), std::abs(sq1 % 8 - sq2 % 8));
}

void Lawliet::initTables() {
    if (tablesInitialized) return;
    tablesInitialized = true;
    const int victimOrder[7] = {0, 100, 320, 330, 500, 900, 20000};
    for (int attacker = 0; attacker < 6; ++attacker)
        for (int victim = 0; victim < 7; ++victim)
            mvvLva[attacker][victim] = victimOrder[victim] * 10 - attacker;

    for (int f = 0; f < 8; ++f) {
        fileMasks[f] = 0;
        for (int r = 0; r < 8; ++r) fileMasks[f] |= (1ULL << (r * 8 + f));
    }
    for (int r = 0; r < 8; ++r) {
        rankMasks[r] = 0;
        for (int f = 0; f < 8; ++f) rankMasks[r] |= (1ULL << (r * 8 + f));
    }

    for (int sq = 0; sq < 64; ++sq) {
        int r = sq / 8, f = sq % 8;
        pawnIsolatedMask[sq] = 0;
        if (f > 0) pawnIsolatedMask[sq] |= fileMasks[f - 1];
        if (f < 7) pawnIsolatedMask[sq] |= fileMasks[f + 1];

        pawnPassedMask[0][sq] = 0; pawnPassedMask[1][sq] = 0;
        for (int curR = 0; curR < 8; ++curR) {
            for (int curF = std::max(0, f - 1); curF <= std::min(7, f + 1); ++curF) {
                int curSq = curR * 8 + curF;
                if (curR < r) pawnPassedMask[0][sq] |= (1ULL << curSq);
                if (curR > r) pawnPassedMask[1][sq] |= (1ULL << curSq);
            }
        }
    }

    for (int d = 0; d < 128; ++d) {
        for (int m = 0; m < 256; ++m) {
            if (d == 0 || m == 0) {
                lmrTable[d][m] = 0;
            } else {
                lmrTable[d][m] = static_cast<int>(0.5 + std::log(d) * std::log(m) / 2.0);
            }
        }
    }
}

int Lawliet::encodeCastling(const Board& board) { return (board.castleWK ? 1 : 0) | (board.castleWQ ? 2 : 0) | (board.castleBK ? 4 : 0) | (board.castleBQ ? 8 : 0); }
int Lawliet::scoreToTT(int score, int ply) { return score > INF - 1000 ? score + ply : (score < -INF + 1000 ? score - ply : score); }
int Lawliet::scoreFromTT(int score, int ply) { return score > INF - 1000 ? score - ply : (score < -INF + 1000 ? score + ply : score); }
void Lawliet::setDepth(int depth) { maxDepth = depth; }
void Lawliet::reset() { for (auto& entry : transpositionTable) entry = TTEntry{}; }

uint64_t Lawliet::computeHash(const Board& board) const {
    uint64_t hash = 0;
    for (int i = 0; i < 12; ++i) {
        uint64_t bb = board.pieceBB[i];
        while (bb) { hash ^= zobristPiece[__builtin_ctzll(bb)][i]; bb &= bb - 1; }
    }
    hash ^= zobristCastle[encodeCastling(board)];
    if (board.enPassantTarget >= 0) hash ^= zobristEp[board.enPassantTarget];
    if (board.turn == Board::BLACK) hash ^= zobristSide;
    return hash;
}

uint64_t Lawliet::seeAttackers(const Board& board, int sq, uint64_t occupied) const {
    uint64_t attackers = 0;
    attackers |= (Board::pawnAttacks[0][sq] & board.pieceBB[6]);
    attackers |= (Board::pawnAttacks[1][sq] & board.pieceBB[0]);
    attackers |= (Board::knightAttacks[sq] & (board.pieceBB[1] | board.pieceBB[7]));
    attackers |= (Board::kingAttacks[sq] & (board.pieceBB[5] | board.pieceBB[11]));
    attackers |= (Board::getBishopAttacks(sq, occupied) & (board.pieceBB[2] | board.pieceBB[8] | board.pieceBB[4] | board.pieceBB[10]));
    attackers |= (Board::getRookAttacks(sq, occupied) & (board.pieceBB[3] | board.pieceBB[9] | board.pieceBB[4] | board.pieceBB[10]));
    return attackers;
}

uint64_t Lawliet::seeXrays(const Board& board, int sq, uint64_t occupied) const {
    uint64_t xrays = 0;
    xrays |= (Board::getBishopAttacks(sq, occupied) & (board.pieceBB[2] | board.pieceBB[8] | board.pieceBB[4] | board.pieceBB[10]));
    xrays |= (Board::getRookAttacks(sq, occupied) & (board.pieceBB[3] | board.pieceBB[9] | board.pieceBB[4] | board.pieceBB[10]));
    return xrays & occupied;
}

int Lawliet::see(const Board& board, int from, int to) const {
    int sq = to, piece = std::abs(board.getPiece(from)), victim = std::abs(board.getPiece(to));
    if (piece == 1 && to == board.enPassantTarget) victim = 1;
    if (piece == 0) return 0;
    int gain[32], d = 0;
    gain[d] = victim == 0 ? 0 : Board::pieceValuesMidgame[victim - 1];
    uint64_t occupied = board.occ, attackers = seeAttackers(board, sq, occupied);
    occupied &= ~(1ULL << from); attackers &= ~(1ULL << from); attackers |= seeXrays(board, sq, occupied);
    int side = (board.getPiece(from) > 0) ? 0 : 1, currentPieceValue = Board::pieceValuesMidgame[piece - 1];
    while (true) {
        if (d >= 31) break;
        uint64_t myAttackers = attackers & board.colorBB[side];
        if (!myAttackers) break;
        int cheapestSq = -1, cheapestVal = INF;
        for (int pType = 0; pType < 6; ++pType) {
            uint64_t subset = myAttackers & board.pieceBB[side * 6 + pType];
            if (subset) { cheapestSq = __builtin_ctzll(subset); cheapestVal = Board::pieceValuesMidgame[pType]; break; }
        }
        if (cheapestSq == -1) break;
        d++; gain[d] = currentPieceValue - gain[d - 1]; currentPieceValue = cheapestVal;
        occupied &= ~(1ULL << cheapestSq); attackers &= ~(1ULL << cheapestSq); attackers |= seeXrays(board, sq, occupied);
        side = 1 - side;
    }
    while (d > 0) { gain[d - 1] = std::max(-gain[d], gain[d - 1]); d--; }
    return gain[0];
}

uint64_t Lawliet::getPinnedPieces(const Board& board, int kingSq, int friendlyColor) const {
    if (kingSq == -1) return 0;
    uint64_t pinned = 0;
    int friendlyColorIdx = (friendlyColor == Board::WHITE) ? 0 : 1, enemyColorIdx = 1 - friendlyColorIdx;
    uint64_t enemySliders = board.pieceBB[enemyColorIdx * 6 + 4];
    uint64_t enemyDiagonalSliders = enemySliders | board.pieceBB[enemyColorIdx * 6 + 2];
    uint64_t bishopPinners = Board::getBishopAttacks(kingSq, board.colorBB[enemyColorIdx]) & enemyDiagonalSliders;
    while (bishopPinners) {
        int pinnerSq = __builtin_ctzll(bishopPinners); bishopPinners &= bishopPinners - 1;
        uint64_t ray = Board::getBishopAttacks(kingSq, board.occ) & Board::getBishopAttacks(pinnerSq, board.occ);
        uint64_t friendlyOnRay = ray & board.colorBB[friendlyColorIdx];
        if (__builtin_popcountll(friendlyOnRay) == 1) pinned |= friendlyOnRay;
    }
    uint64_t enemyStraightSliders = enemySliders | board.pieceBB[enemyColorIdx * 6 + 3];
    uint64_t rookPinners = Board::getRookAttacks(kingSq, board.colorBB[enemyColorIdx]) & enemyStraightSliders;
    while (rookPinners) {
        int pinnerSq = __builtin_ctzll(rookPinners); rookPinners &= rookPinners - 1;
        uint64_t ray = Board::getRookAttacks(kingSq, board.occ) & Board::getRookAttacks(pinnerSq, board.occ);
        uint64_t friendlyOnRay = ray & board.colorBB[friendlyColorIdx];
        if (__builtin_popcountll(friendlyOnRay) == 1) pinned |= friendlyOnRay;
    }
    return pinned;
}

// ============================================================================
// DYNAMIC TAPERED EVALUATION ENGINE
// ============================================================================
int Lawliet::evaluateBoard(const Board& board, int alpha, int beta, const SearchContext* ctx) const {
    int phase = 0;
    for (int i = 0; i < 12; ++i) {
        int type = i % 6, count = __builtin_popcountll(board.pieceBB[i]);
        if (type == 1 || type == 2) phase += count;
        else if (type == 3) phase += count * 2;
        else if (type == 4) phase += count * 4;
    }
    phase = std::min(24, phase);

    int mgScore = board.mgPst;
    int egScore = board.egPst;

    // Apply Static Evaluation Correction History (CorrHist) dynamically based on pawn hash
    if (ctx) {
        uint64_t pawnKey = 0;
        uint64_t wp = board.pieceBB[0];
        while (wp) {
            pawnKey ^= zobristPiece[__builtin_ctzll(wp)][0];
            wp &= wp - 1;
        }
        uint64_t bp = board.pieceBB[6];
        while (bp) {
            pawnKey ^= zobristPiece[__builtin_ctzll(bp)][6];
            bp &= bp - 1;
        }
        int sideIdx = (board.turn == Board::WHITE) ? 0 : 1;
        int correction = ctx->corrHist[sideIdx][pawnKey % 16384];
        mgScore += correction / 16;
        egScore += correction / 16;
    }

    // Bishop Pair Endgame scaling
    int whiteBishops = __builtin_popcountll(board.pieceBB[2]), blackBishops = __builtin_popcountll(board.pieceBB[8]);
    if (whiteBishops >= 2) { mgScore += 30; egScore += 65; }
    if (blackBishops >= 2) { mgScore -= 30; egScore -= 65; }

    uint64_t whitePawns = board.pieceBB[0], blackPawns = board.pieceBB[6], allPawns = whitePawns | blackPawns;

    uint64_t wPawnAttacks = 0, bPawnAttacks = 0, wp_temp = whitePawns, bp_temp = blackPawns;
    while (wp_temp) { wPawnAttacks |= Board::pawnAttacks[0][__builtin_ctzll(wp_temp)]; wp_temp &= wp_temp - 1; }
    while (bp_temp) { bPawnAttacks |= Board::pawnAttacks[1][__builtin_ctzll(bp_temp)]; bp_temp &= bp_temp - 1; }

    uint64_t wSafe = ~bPawnAttacks;
    uint64_t bSafe = ~wPawnAttacks;

    // Upgraded passed pawn scaling
    const int passedPawnMg[8] = { 0, 150, 100, 60, 30, 15, 5, 0 };
    const int passedPawnEg[8] = { 0, 320, 240, 150, 90, 40, 15, 0 };

    int wkSq = board.findKing(Board::WHITE), bkSq = board.findKing(Board::BLACK);

    // --- PAWN HASH TABLE (PHT) SKELETON LOOKUP ---
    uint64_t pawnKey = 0;
    uint64_t wp_key = whitePawns;
    while (wp_key) {
        pawnKey ^= zobristPiece[__builtin_ctzll(wp_key)][0];
        wp_key &= wp_key - 1;
    }
    uint64_t bp_key = blackPawns;
    while (bp_key) {
        pawnKey ^= zobristPiece[__builtin_ctzll(bp_key)][6];
        bp_key &= bp_key - 1;
    }

    PawnEntry& pEntry = pawnTable[pawnKey & (PAWN_SIZE - 1)];

    int pawnMg = 0;
    int pawnEg = 0;

    if (pEntry.key == pawnKey) {
        // Retrieve highly optimal cached pawn scores
        pawnMg = pEntry.mgScore;
        pawnEg = pEntry.egScore;
    } else {
        // Reset dynamic bitboards inside PHT entry
        pEntry.wPassedPawns = 0;
        pEntry.bPassedPawns = 0;

        // White Pawns static skeleton calculation
        uint64_t wp = whitePawns;
        while (wp) {
            int sq = __builtin_ctzll(wp); wp &= wp - 1; int file = sq % 8, rank = sq / 8;

            if (__builtin_popcountll(whitePawns & fileMasks[file]) > 1) { pawnMg -= 20; pawnEg -= 25; }
            if (!(whitePawns & pawnIsolatedMask[sq])) { pawnMg -= 20; pawnEg -= 25; }
            if (rank >= 2 && rank <= 5) {
                uint64_t ranksBehind = ~((1ULL << (rank * 8)) - 1);
                if (!(pawnIsolatedMask[sq] & whitePawns & ranksBehind) && (Board::pawnAttacks[0][sq - 8] & blackPawns)) { pawnMg -= 20; pawnEg -= 25; }
            }

            bool connected = false;
            if (file > 0 && (whitePawns & fileMasks[file - 1])) connected = true;
            if (file < 7 && (whitePawns & fileMasks[file + 1])) connected = true;
            if (connected) {
                bool defended = (Board::pawnAttacks[1][sq] & whitePawns);
                uint64_t sameRankNeighbors = 0;
                if (file > 0) sameRankNeighbors |= (1ULL << (sq - 1));
                if (file < 7) sameRankNeighbors |= (1ULL << (sq + 1));
                bool phalanx = (sameRankNeighbors & whitePawns);

                int bonusMg = 5, bonusEg = 5;
                if (defended) { bonusMg += 10; bonusEg += 15; }
                if (phalanx) { bonusMg += 8; bonusEg += 10; }
                pawnMg += bonusMg; pawnEg += bonusEg;
            }

            if (!(blackPawns & pawnPassedMask[0][sq])) {
                pEntry.wPassedPawns |= (1ULL << sq); // Store white passed pawn
                int bonusMg = passedPawnMg[rank];
                int bonusEg = passedPawnEg[rank];

                // Blocked passed pawn protection scaling
                int frontSq = sq - 8;
                if (frontSq >= 0) {
                    int frontPiece = board.getPiece(frontSq);
                    if (frontPiece != 0) {
                        if (frontPiece < 0) { // Blocked by enemy piece
                            bonusMg /= 2;
                            bonusEg /= 2;
                        } else { // Blocked by friendly piece
                            bonusMg = (bonusMg * 3) / 4;
                            bonusEg = (bonusEg * 3) / 4;
                        }
                    }
                }

                pawnMg += bonusMg; pawnEg += bonusEg;

                // Connected passed pawns bonus (purely static pawn configuration)
                bool connectedPasser = false;
                uint64_t adjacentFiles = pawnIsolatedMask[sq];
                uint64_t otherPassers = adjacentFiles & whitePawns;
                while (otherPassers) {
                    int opSq = __builtin_ctzll(otherPassers); otherPassers &= otherPassers - 1;
                    if (!(blackPawns & pawnPassedMask[0][opSq])) {
                        connectedPasser = true;
                        break;
                    }
                }
                if (connectedPasser) {
                    pawnMg += 25 + rank * 5;
                    pawnEg += 50 + rank * 15;
                }
            }
        }

        // Black Pawns static skeleton calculation
        uint64_t bp = blackPawns;
        while (bp) {
            int sq = __builtin_ctzll(bp); bp &= bp - 1; int file = sq % 8, rank = sq / 8;
            if (__builtin_popcountll(blackPawns & fileMasks[file]) > 1) { pawnMg += 20; pawnEg += 25; }
            if (!(blackPawns & pawnIsolatedMask[sq])) { pawnMg += 20; pawnEg += 25; }
            if (rank >= 2 && rank <= 5) {
                uint64_t ranksBehind = (1ULL << (rank * 8)) - 1;
                if (!(pawnIsolatedMask[sq] & blackPawns & ranksBehind) && (Board::pawnAttacks[1][sq + 8] & whitePawns)) { pawnMg += 20; pawnEg += 25; }
            }

            bool connected = false;
            if (file > 0 && (blackPawns & fileMasks[file - 1])) connected = true;
            if (file < 7 && (blackPawns & fileMasks[file + 1])) connected = true;
            if (connected) {
                bool defended = (Board::pawnAttacks[0][sq] & blackPawns);
                uint64_t sameRankNeighbors = 0;
                if (file > 0) sameRankNeighbors |= (1ULL << (sq - 1));
                if (file < 7) sameRankNeighbors |= (1ULL << (sq + 1));
                bool phalanx = (sameRankNeighbors & blackPawns);

                int bonusMg = 5, bonusEg = 5;
                if (defended) { bonusMg += 10; bonusEg += 15; }
                if (phalanx) { bonusMg += 8; bonusEg += 10; }
                pawnMg -= bonusMg; pawnEg -= bonusEg;
            }

            if (!(whitePawns & pawnPassedMask[1][sq])) {
                pEntry.bPassedPawns |= (1ULL << sq); // Store black passed pawn
                int rMapped = 7 - rank;
                int bonusMg = passedPawnMg[rMapped];
                int bonusEg = passedPawnEg[rMapped];

                // Blocked passed pawn protection scaling
                int frontSq = sq + 8;
                if (frontSq < 64) {
                    int frontPiece = board.getPiece(frontSq);
                    if (frontPiece != 0) {
                        if (frontPiece > 0) { // Blocked by enemy piece
                            bonusMg /= 2;
                            bonusEg /= 2;
                        } else { // Blocked by friendly piece
                            bonusMg = (bonusMg * 3) / 4;
                            bonusEg = (bonusEg * 3) / 4;
                        }
                    }
                }

                pawnMg -= bonusMg; pawnEg -= bonusEg;

                // Connected passed pawns bonus (purely static pawn configuration)
                bool connectedPasser = false;
                uint64_t adjacentFiles = pawnIsolatedMask[sq];
                uint64_t otherPassers = adjacentFiles & blackPawns;
                while (otherPassers) {
                    int opSq = __builtin_ctzll(otherPassers); otherPassers &= otherPassers - 1;
                    if (!(whitePawns & pawnPassedMask[1][opSq])) {
                        connectedPasser = true;
                        break;
                    }
                }
                if (connectedPasser) {
                    int rMapped = 7 - rank;
                    pawnMg -= (25 + rMapped * 5);
                    pawnEg -= (50 + rMapped * 15);
                }
            }
        }

        // Cache the completed static skeleton calculations
        pEntry.key = pawnKey;
        pEntry.mgScore = pawnMg;
        pEntry.egScore = pawnEg;
    }

    mgScore += pawnMg;
    egScore += pawnEg;

    // --- DYNAMIC PASSED PAWN BONUSES (Outside PHT to keep king and rooks dynamic) ---
    uint64_t wPassed = pEntry.wPassedPawns;
    while (wPassed) {
        int sq = __builtin_ctzll(wPassed); wPassed &= wPassed - 1;
        int file = sq % 8, rank = sq / 8;
        uint64_t behindMask = fileMasks[file] & ~((1ULL << ((rank + 1) * 8)) - 1);
        if (board.pieceBB[3] & behindMask) { mgScore += 20; egScore += 35; }
        if (board.pieceBB[9] & behindMask) { mgScore -= 10; egScore -= 20; }

        // Dynamic King proximity bonus for White passed pawn
        if (wkSq != -1 && bkSq != -1) {
            int wDist = kingDistance(wkSq, sq);
            int bDist = kingDistance(bkSq, sq);
            egScore += (bDist - wDist) * 25;
        }
    }

    uint64_t bPassed = pEntry.bPassedPawns;
    while (bPassed) {
        int sq = __builtin_ctzll(bPassed); bPassed &= bPassed - 1;
        int file = sq % 8, rank = sq / 8;
        uint64_t behindMask = fileMasks[file] & ((1ULL << (rank * 8)) - 1);
        if (board.pieceBB[9] & behindMask) { mgScore -= 20; egScore -= 35; }
        if (board.pieceBB[3] & behindMask) { mgScore += 10; egScore += 20; }

        // Dynamic King proximity bonus for Black passed pawn
        if (wkSq != -1 && bkSq != -1) {
            int wDist = kingDistance(wkSq, sq);
            int bDist = kingDistance(bkSq, sq);
            egScore -= (wDist - bDist) * 25;
        }
    }

    // Center Space Control Pawn Occupation Heuristic
    if (board.getPiece(35) == 1)  { mgScore += 15; } // White Pawn on d4
    if (board.getPiece(36) == 1)  { mgScore += 15; } // White Pawn on e4
    if (board.getPiece(27) == -1) { mgScore -= 15; } // Black Pawn on d5
    if (board.getPiece(28) == -1) { mgScore -= 15; } // Black Pawn on e5

    mgScore += __builtin_popcountll((rankMasks[2]|rankMasks[3]|rankMasks[4]) & wPawnAttacks & ~blackPawns & ~bPawnAttacks) * 4;
    mgScore -= __builtin_popcountll((rankMasks[3]|rankMasks[4]|rankMasks[5]) & bPawnAttacks & ~whitePawns & ~wPawnAttacks) * 4;

    for (int pType = 1; pType <= 5; ++pType) {
        uint64_t bb = board.pieceBB[pType];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            if (Board::pawnAttacks[0][sq] & blackPawns) {
                int pmg = (pType == 4) ? 100 : (pType == 3 ? 60 : 40), peg = (pType == 4) ? 80 : (pType == 3 ? 45 : 30);
                mgScore -= pmg; egScore -= peg;
            }
        }
        bb = board.pieceBB[pType + 6];
        while (bb) {
            int sq = __builtin_ctzll(bb); bb &= bb - 1;
            if (Board::pawnAttacks[1][sq] & whitePawns) {
                int pmg = (pType == 4) ? 100 : (pType == 3 ? 60 : 40), peg = (pType == 4) ? 80 : (pType == 3 ? 45 : 30);
                mgScore += pmg; egScore += peg;
            }
        }
    }

    // Doubled rooks on files
    for (int f = 0; f < 8; ++f) {
        uint64_t fileMask = fileMasks[f];
        int wRooks = __builtin_popcountll(board.pieceBB[3] & fileMask);
        if (wRooks >= 2) { mgScore += 16; egScore += 24; }
        int bRooks = __builtin_popcountll(board.pieceBB[9] & fileMask);
        if (bRooks >= 2) { mgScore -= 16; egScore -= 24; }
    }

    // Advanced 7th/2nd rank Rook activity and doubled rooks on 7th/2nd rank
    int wRooksOn7th = __builtin_popcountll(board.pieceBB[3] & rankMasks[1]);
    if (wRooksOn7th > 0) {
        bool target = (board.pieceBB[11] & rankMasks[0]) || (board.pieceBB[6] & rankMasks[1]);
        int bonusMg = target ? 45 : 20;
        int bonusEg = target ? 60 : 30;
        mgScore += bonusMg * wRooksOn7th;
        egScore += bonusEg * wRooksOn7th;
        if (wRooksOn7th >= 2) { mgScore += 35; egScore += 50; }
    }
    int bRooksOn2nd = __builtin_popcountll(board.pieceBB[9] & rankMasks[6]);
    if (bRooksOn2nd > 0) {
        bool target = (board.pieceBB[5] & rankMasks[7]) || (board.pieceBB[0] & rankMasks[6]);
        int bonusMg = target ? 45 : 20;
        int bonusEg = target ? 60 : 30;
        mgScore -= bonusMg * bRooksOn2nd;
        egScore -= bonusEg * bRooksOn2nd;
        if (bRooksOn2nd >= 2) { mgScore -= 35; egScore -= 50; }
    }

    uint64_t wr_pos = board.pieceBB[3];
    while (wr_pos) {
        int sq = __builtin_ctzll(wr_pos); wr_pos &= wr_pos - 1; uint64_t fMask = fileMasks[sq % 8];
        if (!(allPawns & fMask)) { mgScore += 20; egScore += 30; }
        else if (!(whitePawns & fMask)) { mgScore += 10; egScore += 15; }
    }
    uint64_t br_pos = board.pieceBB[9];
    while (br_pos) {
        int sq = __builtin_ctzll(br_pos); br_pos &= br_pos - 1; uint64_t fMask = fileMasks[sq % 8];
        if (!(allPawns & fMask)) { mgScore -= 20; egScore -= 30; }
        else if (!(blackPawns & fMask)) { mgScore -= 10; egScore -= 15; }
    }

    if (phase >= 12) {
        int wUnd = 0, bUnd = 0;
        if (board.getPiece(57) == 2) wUnd++; if (board.getPiece(58) == 3) wUnd++; if (board.getPiece(61) == 3) wUnd++; if (board.getPiece(62) == 2) wUnd++;
        mgScore -= wUnd * 15;
        if (board.getPiece(1) == -2) bUnd++; if (board.getPiece(2) == -3) bUnd++; if (board.getPiece(5) == -3) bUnd++; if (board.getPiece(6) == -2) bUnd++;
        mgScore += bUnd * 15;

        uint64_t wq_pieces = board.pieceBB[4];
        if (wq_pieces && __builtin_ctzll(wq_pieces) != 59) mgScore -= wUnd * 20;
        uint64_t bq_pieces = board.pieceBB[10];
        if (bq_pieces && __builtin_ctzll(bq_pieces) != 3) mgScore += bUnd * 20;
    }

    // --- Dynamic Tapered Safe Mobility (Pinned check removed for high performance) ---
    uint64_t wn_mob = board.pieceBB[1];
    while (wn_mob) {
        int sq = __builtin_ctzll(wn_mob); wn_mob &= wn_mob - 1;
        int c = __builtin_popcountll(Board::knightAttacks[sq] & ~board.colorBB[0] & wSafe);
        mgScore += c * 4; egScore += c * 4;
    }
    uint64_t wb_mob = board.pieceBB[2];
    while (wb_mob) {
        int sq = __builtin_ctzll(wb_mob); wb_mob &= wb_mob - 1;
        int c = __builtin_popcountll(board.getBishopAttacks(sq, board.occ) & ~board.colorBB[0] & wSafe);
        mgScore += c * 3; egScore += c * 3;
    }
    uint64_t wr_mob = board.pieceBB[3];
    while (wr_mob) {
        int sq = __builtin_ctzll(wr_mob); wr_mob &= wr_mob - 1;
        int c = __builtin_popcountll(board.getRookAttacks(sq, board.occ) & ~board.colorBB[0] & wSafe);
        mgScore += c * 2; egScore += c * 4;
    }
    uint64_t wq_mob = board.pieceBB[4];
    while (wq_mob) {
        int sq = __builtin_ctzll(wq_mob); wq_mob &= wq_mob - 1;
        uint64_t attacks = board.getBishopAttacks(sq, board.occ) | board.getRookAttacks(sq, board.occ);
        int c = __builtin_popcountll(attacks & ~board.colorBB[0] & wSafe);
        mgScore += c * 1; egScore += c * 2;
    }

    uint64_t bn_mob = board.pieceBB[7];
    while (bn_mob) {
        int sq = __builtin_ctzll(bn_mob); bn_mob &= bn_mob - 1;
        int c = __builtin_popcountll(Board::knightAttacks[sq] & ~board.colorBB[1] & bSafe);
        mgScore -= c * 4; egScore -= c * 4;
    }
    uint64_t bb_mob = board.pieceBB[8];
    while (bb_mob) {
        int sq = __builtin_ctzll(bb_mob); bb_mob &= bb_mob - 1;
        int c = __builtin_popcountll(board.getBishopAttacks(sq, board.occ) & ~board.colorBB[1] & bSafe);
        mgScore -= c * 3; egScore -= c * 3;
    }
    uint64_t br_mob = board.pieceBB[9];
    while (br_mob) {
        int sq = __builtin_ctzll(br_mob); br_mob &= br_mob - 1;
        int c = __builtin_popcountll(Board::getRookAttacks(sq, board.occ) & ~board.colorBB[1] & bSafe);
        mgScore -= c * 2; egScore -= c * 4;
    }
    uint64_t bq_mob = board.pieceBB[10];
    while (bq_mob) {
        int sq = __builtin_ctzll(bq_mob); bq_mob &= bq_mob - 1;
        uint64_t attacks = board.getBishopAttacks(sq, board.occ) | board.getRookAttacks(sq, board.occ);
        int c = __builtin_popcountll(attacks & ~board.colorBB[1] & bSafe);
        mgScore -= c * 1; egScore -= c * 2;
    }

    uint64_t w_minors = board.pieceBB[1] | board.pieceBB[2]; while (w_minors) { int sq = __builtin_ctzll(w_minors); w_minors &= w_minors - 1; if (Board::pawnAttacks[1][sq] & whitePawns) { mgScore += 15; egScore += 10; } }
    uint64_t b_minors = board.pieceBB[7] | board.pieceBB[8]; while (b_minors) { int sq = __builtin_ctzll(b_minors); b_minors &= b_minors - 1; if (Board::pawnAttacks[0][sq] & blackPawns) { mgScore -= 15; egScore -= 10; } }

    // Knight and Bishop Outposts
    uint64_t wn_pos_out = board.pieceBB[1];
    while (wn_pos_out) {
        int sq = __builtin_ctzll(wn_pos_out); wn_pos_out &= wn_pos_out - 1;
        int rank = sq / 8, file = sq % 8;
        if (rank >= 2 && rank <= 5) {
            bool defendedByPawn = (Board::pawnAttacks[1][sq] & whitePawns);
            if (defendedByPawn) {
                bool attackable = false;
                if (file > 0 && (sq - 9 >= 0) && (board.getPiece(sq - 9) == -1)) attackable = true;
                if (file < 7 && (sq - 7 >= 0) && (board.getPiece(sq - 7) == -1)) attackable = true;
                if (!attackable) { mgScore += 25; egScore += 15; }
                else { mgScore += 10; egScore += 5; }
            }
        }
    }
    uint64_t wb_pos_out = board.pieceBB[2];
    while (wb_pos_out) {
        int sq = __builtin_ctzll(wb_pos_out); wb_pos_out &= wb_pos_out - 1;
        int rank = sq / 8, file = sq % 8;
        if (rank >= 2 && rank <= 5) {
            bool defendedByPawn = (Board::pawnAttacks[1][sq] & whitePawns);
            if (defendedByPawn) {
                bool attackable = false;
                if (file > 0 && (sq - 9 >= 0) && (board.getPiece(sq - 9) == -1)) attackable = true;
                if (file < 7 && (sq - 7 >= 0) && (board.getPiece(sq - 7) == -1)) attackable = true;
                if (!attackable) { mgScore += 15; egScore += 10; }
            }
        }
    }
    uint64_t bn_pos_out = board.pieceBB[7];
    while (bn_pos_out) {
        int sq = __builtin_ctzll(bn_pos_out); bn_pos_out &= bn_pos_out - 1;
        int rank = sq / 8, file = sq % 8;
        if (rank >= 2 && rank <= 5) {
            bool defendedByPawn = (Board::pawnAttacks[0][sq] & blackPawns);
            if (defendedByPawn) {
                bool attackable = false;
                if (file > 0 && (sq + 7 < 64) && (board.getPiece(sq + 7) == 1)) attackable = true;
                if (file < 7 && (sq + 9 < 64) && (board.getPiece(sq + 9) == 1)) attackable = true;
                if (!attackable) { mgScore -= 25; egScore -= 15; }
                else { mgScore -= 10; egScore -= 5; }
            }
        }
    }
    uint64_t bb_pos_out = board.pieceBB[8];
    while (bb_pos_out) {
        int sq = __builtin_ctzll(bb_pos_out); bb_pos_out &= bb_pos_out - 1;
        int rank = sq / 8, file = sq % 8;
        if (rank >= 2 && rank <= 5) {
            bool defendedByPawn = (Board::pawnAttacks[0][sq] & blackPawns);
            if (defendedByPawn) {
                bool attackable = false;
                if (file > 0 && (sq + 7 < 64) && (board.getPiece(sq + 7) == 1)) attackable = true;
                if (file < 7 && (sq + 9 < 64) && (board.getPiece(sq + 9) == 1)) attackable = true;
                if (!attackable) { mgScore -= 15; egScore -= 10; }
            }
        }
    }

    // Undefended minor pieces penalties
    uint64_t w_minors_undef = board.pieceBB[1] | board.pieceBB[2];
    while (w_minors_undef) {
        int sq = __builtin_ctzll(w_minors_undef); w_minors_undef &= w_minors_undef - 1;
        if (!(Board::pawnAttacks[1][sq] & whitePawns)) { mgScore -= 12; egScore -= 8; }
    }
    uint64_t b_minors_undef = board.pieceBB[7] | board.pieceBB[8];
    while (b_minors_undef) {
        int sq = __builtin_ctzll(b_minors_undef); b_minors_undef &= b_minors_undef - 1;
        if (!(Board::pawnAttacks[0][sq] & blackPawns)) { mgScore += 12; egScore += 8; }
    }

    // King centralization in endgames
    if (wkSq != -1) {
        int r = wkSq / 8, f = wkSq % 8;
        int centerDist = std::abs(r - 3) + std::abs(f - 3);
        egScore += (8 - centerDist) * 5;
    }
    if (bkSq != -1) {
        int r = bkSq / 8, f = bkSq % 8;
        int centerDist = std::abs(r - 3) + std::abs(f - 3);
        egScore -= (8 - centerDist) * 5;
    }

    if (wkSq != -1) {
        int kf = wkSq % 8;
        uint64_t blackMajors = board.pieceBB[9] | board.pieceBB[10];

        for (int file = std::max(0, kf - 1); file <= std::min(7, kf + 1); ++file) {
            uint64_t filePawns = board.pieceBB[0] & fileMasks[file];
            if (!filePawns) {
                bool open = !(board.pieceBB[6] & fileMasks[file]);
                int penMg = open ? 60 : 40;
                int penEg = open ? 20 : 15;
                if (blackMajors & fileMasks[file]) {
                    penMg += 25; penEg += 10;
                }
                mgScore -= penMg; egScore -= penEg;
            } else {
                int closestPawnSq = 63 - __builtin_clzll(filePawns);
                int r = closestPawnSq / 8;
                int penMg = 0, penEg = 0;
                if (r == 6) { penMg = 0; penEg = 0; }
                else if (r == 5) { penMg = 10; penEg = 5; }
                else if (r == 4) { penMg = 25; penEg = 10; }
                else if (r <= 3) { penMg = 35; penEg = 15; }
                mgScore -= penMg; egScore -= penEg;
            }
        }

        for (int file = std::max(0, kf - 1); file <= std::min(7, kf + 1); ++file) {
            uint64_t enemyPawns = board.pieceBB[6] & fileMasks[file];
            if (enemyPawns) {
                int closestEnemyPawnSq = 63 - __builtin_clzll(enemyPawns);
                int r = closestEnemyPawnSq / 8;
                if (r >= 4) {
                    int stormPen = (r == 4) ? 10 : ((r == 5) ? 25 : 50);
                    mgScore -= stormPen;
                }
            }
        }

        int r = wkSq / 8;
        if (r < 6) mgScore -= (6 - r) * 25;
    } else {
        mgScore -= 120;
    }

    if (bkSq != -1) {
        int kf = bkSq % 8;
        uint64_t whiteMajors = board.pieceBB[3] | board.pieceBB[4];

        for (int file = std::max(0, kf - 1); file <= std::min(7, kf + 1); ++file) {
            uint64_t filePawns = board.pieceBB[6] & fileMasks[file];
            if (!filePawns) {
                bool open = !(board.pieceBB[0] & fileMasks[file]);
                int penMg = open ? 60 : 40;
                int penEg = open ? 20 : 15;
                if (whiteMajors & fileMasks[file]) {
                    penMg += 25; penEg += 10;
                }
                mgScore += penMg; egScore += penEg;
            } else {
                int closestPawnSq = __builtin_ctzll(filePawns);
                int r = closestPawnSq / 8;
                int penMg = 0, penEg = 0;
                if (r == 1) { penMg = 0; penEg = 0; }
                else if (r == 2) { penMg = 10; penEg = 5; }
                else if (r == 3) { penMg = 25; penEg = 10; }
                else if (r >= 4) { penMg = 35; penEg = 15; }
                mgScore += penMg; egScore += penEg;
            }
        }

        for (int file = std::max(0, kf - 1); file <= std::min(7, kf + 1); ++file) {
            uint64_t enemyPawns = board.pieceBB[0] & fileMasks[file];
            if (enemyPawns) {
                int closestEnemyPawnSq = __builtin_ctzll(enemyPawns);
                int r = closestEnemyPawnSq / 8;
                if (r <= 3) {
                    int stormPen = (r == 3) ? 10 : ((r == 2) ? 25 : 50);
                    mgScore += stormPen;
                }
            }
        }

        int r = bkSq / 8;
        if (r > 1) mgScore += (r - 1) * 25;
    } else {
        mgScore += 120;
    }

    uint64_t wKingZone = 0; if (wkSq != -1) { wKingZone = Board::kingAttacks[wkSq] | (1ULL << wkSq); if (wkSq / 8 > 0) wKingZone |= (Board::kingAttacks[wkSq] >> 8); }
    uint64_t bKingZone = 0; if (bkSq != -1) { bKingZone = Board::kingAttacks[bkSq] | (1ULL << bkSq); if (bkSq / 8 < 7) bKingZone |= (Board::kingAttacks[bkSq] << 8); }

    int wKingDanger = 0, bKingDanger = 0;
    if (phase >= 6) {
        if (wkSq != -1) {
            uint64_t bn = board.pieceBB[7];
            while (bn) { int sq = __builtin_ctzll(bn); bn &= bn - 1; if (Board::knightAttacks[sq] & wKingZone) wKingDanger += 2; }
            uint64_t bb = board.pieceBB[8];
            while (bb) { int sq = __builtin_ctzll(bb); bb &= bb - 1; if (board.getBishopAttacks(sq, board.occ) & wKingZone) wKingDanger += 2; }
            uint64_t br = board.pieceBB[9];
            while (br) { int sq = __builtin_ctzll(br); br &= br - 1; if (Board::getRookAttacks(sq, board.occ) & wKingZone) wKingDanger += 3; }
            uint64_t bq = board.pieceBB[10];
            while (bq) { int sq = __builtin_ctzll(bq); bq &= bq - 1; uint64_t q_attacks = board.getBishopAttacks(sq, board.occ) | board.getRookAttacks(sq, board.occ); if (q_attacks & wKingZone) wKingDanger += 4; }
            if (wKingDanger > 2) mgScore -= (wKingDanger * wKingDanger) * 2;
        }
        if (bkSq != -1) {
            uint64_t wn = board.pieceBB[1];
            while (wn) { int sq = __builtin_ctzll(wn); wn &= wn - 1; if (Board::knightAttacks[sq] & bKingZone) bKingDanger += 2; }
            uint64_t wb = board.pieceBB[2];
            while (wb) { int sq = __builtin_ctzll(wb); wb &= wb - 1; if (board.getBishopAttacks(sq, board.occ) & bKingZone) bKingDanger += 2; }
            uint64_t wr = board.pieceBB[3];
            while (wr) { int sq = __builtin_ctzll(wr); wr &= wr - 1; if (Board::getRookAttacks(sq, board.occ) & bKingZone) bKingDanger += 3; }
            uint64_t wq = board.pieceBB[4];
            while (wq) { int sq = __builtin_ctzll(wq); wq &= wq - 1; uint64_t q_attacks = board.getBishopAttacks(sq, board.occ) | board.getRookAttacks(sq, board.occ); if (q_attacks & bKingZone) bKingDanger += 4; }
            if (bKingDanger > 2) mgScore += (bKingDanger * bKingDanger) * 2;
        }
    }

    // --- Positional planning, development and castling rights ---
    if (phase >= 12) {
        if (board.castleWK) mgScore += 15;
        if (board.castleWQ) mgScore += 10;
        if (board.castleBK) mgScore -= 15;
        if (board.castleBQ) mgScore -= 10;
    }

    if (phase >= 16) {
        if (board.getPiece(57) == 2) mgScore -= 20; // White Knight on b1
        if (board.getPiece(62) == 2) mgScore -= 20; // White Knight on g1
        if (board.getPiece(58) == 3) mgScore -= 15; // White Bishop on c1
        if (board.getPiece(61) == 3) mgScore -= 15; // White Bishop on f1

        if (board.getPiece(1) == -2) mgScore += 20;  // Black Knight on b8
        if (board.getPiece(6) == -2) mgScore += 20;  // Black Knight on g8
        if (board.getPiece(2) == -3) mgScore += 15;  // Black Bishop on c8
        if (board.getPiece(5) == -3) mgScore += 15;  // Black Bishop on f8
    }

    // --- Opposite-Colored Bishops Endgame Scale Factor ---
    int wMinors = __builtin_popcountll(board.pieceBB[1] | board.pieceBB[2]);
    int bMinors = __builtin_popcountll(board.pieceBB[7] | board.pieceBB[8]);
    int wMajors = __builtin_popcountll(board.pieceBB[3] | board.pieceBB[4]);
    int bMajors = __builtin_popcountll(board.pieceBB[9] | board.pieceBB[10]);

    if (whiteBishops == 1 && blackBishops == 1 && wMinors == 1 && bMinors == 1 && wMajors == 0 && bMajors == 0) {
        int wBisSq = __builtin_ctzll(board.pieceBB[2]);
        int bBisSq = __builtin_ctzll(board.pieceBB[8]);
        bool isOpposite = (((1ULL << wBisSq) & 0x55AA55AA55AA55AAULL) != 0) !=
        (((1ULL << bBisSq) & 0x55AA55AA55AA55AAULL) != 0);
        if (isOpposite) {
            // Scale down the endgame evaluation by 50% to reflect drawish nature
            egScore /= 2;
        }
    }

    int finalScore = ((mgScore * phase) + (egScore * (24 - phase))) / 24;
    int relativeScore = (board.turn == Board::WHITE) ? finalScore : -finalScore;
    return relativeScore + 15; // 15 cp Tempo Bonus
}

void Lawliet::generateLegalMoves(const Board& board, int color, Move* out, int& count) const {
    count = 0; int colorIdx = (color == Board::WHITE) ? 0 : 1; uint64_t myPieces = board.colorBB[colorIdx];
    while (myPieces) {
        int from = __builtin_ctzll(myPieces); myPieces &= myPieces - 1; int piece = board.getPiece(from), type = std::abs(piece) - 1;
        if (type == 0) {
            int dir = (color == Board::WHITE) ? -8 : 8, startRank = (color == Board::WHITE) ? 6 : 1, promoRank = (color == Board::WHITE) ? 0 : 7;
            int to1 = from + dir;
            if (to1 >= 0 && to1 < 64 && board.getPiece(to1) == 0) {
                if (to1 / 8 == promoRank) {
                    for (int promo : {5, 2, 4, 3}) {
                        int promoPiece = (color == Board::WHITE) ? promo : -promo;
                        out[count++] = Move{from, to1, piece, 0, promoPiece, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                    }
                } else {
                    out[count++] = Move{from, to1, piece, 0, 0, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                    if (from / 8 == startRank) {
                        int to2 = from + 2 * dir;
                        if (board.getPiece(to2) == 0) out[count++] = Move{from, to2, piece, 0, 0, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                    }
                }
            }
            uint64_t attacks = board.pawnAttacks[colorIdx][from] & board.colorBB[colorIdx ^ 1];
            while (attacks) {
                int to = __builtin_ctzll(attacks); attacks &= attacks - 1; int captured = board.getPiece(to);
                if (to / 8 == promoRank) {
                    for (int promo : {5, 2, 4, 3}) {
                        int promoPiece = (color == Board::WHITE) ? promo : -promo;
                        out[count++] = Move{from, to, piece, captured, promoPiece, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                    }
                } else {
                    out[count++] = Move{from, to, piece, captured, 0, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                }
            }
            if (board.enPassantTarget != -1 && (board.pawnAttacks[colorIdx][from] & (1ULL << board.enPassantTarget))) {
                int capSq = (color == Board::WHITE) ? board.enPassantTarget + 8 : board.enPassantTarget - 8, capPiece = (color == Board::WHITE) ? -1 : 1;
                out[count++] = Move{from, board.enPassantTarget, piece, capPiece, 0, false, -1, -1, 0, true, capSq, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
            }
            continue;
        }

        uint64_t movesBB = 0;
        if (type == 1) movesBB = board.knightAttacks[from];
        else if (type == 2) movesBB = board.getBishopAttacks(from, board.occ);
        else if (type == 3) movesBB = board.getRookAttacks(from, board.occ);
        else if (type == 4) movesBB = board.getBishopAttacks(from, board.occ) | board.getRookAttacks(from, board.occ);
        else if (type == 5) {
            movesBB = board.kingAttacks[from];
            if (color == Board::WHITE && from == 60) {
                if (board.castleWK && !(board.occ & (1ULL<<61)) && !(board.occ & (1ULL<<62)) && !board.isInCheck(Board::WHITE) && !board.isSquareAttacked(61, Board::BLACK) && !board.isSquareAttacked(62, Board::BLACK))
                    out[count++] = Move{60, 62, 6, 0, 0, true, 63, 61, 4, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                if (board.castleWQ && !(board.occ & (1ULL<<59)) && !(board.occ & (1ULL<<58)) && !(board.occ & (1ULL<<57)) && !board.isInCheck(Board::WHITE) && !board.isSquareAttacked(59, Board::BLACK) && !board.isSquareAttacked(58, Board::BLACK))
                    out[count++] = Move{60, 58, 6, 0, 0, true, 56, 59, 4, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
            } else if (color == Board::BLACK && from == 4) {
                if (board.castleBK && !(board.occ & (1ULL<<5)) && !(board.occ & (1ULL<<6)) && !board.isInCheck(Board::BLACK) && !board.isSquareAttacked(5, Board::WHITE) && !board.isSquareAttacked(6, Board::WHITE))
                    out[count++] = Move{4, 6, -6, 0, 0, true, 7, 5, -4, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                if (board.castleBQ && !(board.occ & (1ULL<<3)) && !(board.occ & (1ULL<<2)) && !(board.occ & (1ULL<<1)) && !board.isInCheck(Board::BLACK) && !board.isSquareAttacked(3, Board::WHITE) && !board.isSquareAttacked(2, Board::WHITE))
                    out[count++] = Move{4, 2, -6, 0, 0, true, 0, 3, -4, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
            }
        }
        movesBB &= ~board.colorBB[colorIdx];
        while (movesBB) {
            int to = __builtin_ctzll(movesBB); movesBB &= movesBB - 1;
            out[count++] = Move{from, to, piece, board.getPiece(to), 0, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
        }
    }
}

void Lawliet::generateCaptures(const Board& board, int color, Move* out, int& count) const {
    count = 0; int colorIdx = (color == Board::WHITE) ? 0 : 1; uint64_t myPieces = board.colorBB[colorIdx], oppPieces = board.colorBB[colorIdx ^ 1];
    while (myPieces) {
        int from = __builtin_ctzll(myPieces); myPieces &= myPieces - 1; int piece = board.getPiece(from), type = std::abs(piece) - 1;
        if (type == 0) {
            int dir = (color == Board::WHITE) ? -8 : 8, promoRank = (color == Board::WHITE) ? 0 : 7;
            uint64_t attacks = board.pawnAttacks[colorIdx][from] & oppPieces;
            while (attacks) {
                int to = __builtin_ctzll(attacks); attacks &= attacks - 1; int captured = board.getPiece(to);
                if (to / 8 == promoRank) {
                    for (int promo : {5, 2, 4, 3}) {
                        int promoPiece = (color == Board::WHITE) ? promo : -promo;
                        out[count++] = Move{from, to, piece, captured, promoPiece, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                    }
                } else {
                    out[count++] = Move{from, to, piece, captured, 0, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                }
            }
            int to1 = from + dir;
            if (to1 >= 0 && to1 < 64 && board.getPiece(to1) == 0 && (to1 / 8 == promoRank)) {
                for (int promo : {5, 2, 4, 3}) {
                    int promoPiece = (color == Board::WHITE) ? promo : -promo;
                    out[count++] = Move{from, to1, piece, 0, promoPiece, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
                }
            }
            if (board.enPassantTarget != -1 && (board.pawnAttacks[colorIdx][from] & (1ULL << board.enPassantTarget))) {
                int capSq = (color == Board::WHITE) ? board.enPassantTarget + 8 : board.enPassantTarget - 8, capPiece = (color == Board::WHITE) ? -1 : 1;
                out[count++] = Move{from, board.enPassantTarget, piece, capPiece, 0, false, -1, -1, 0, true, capSq, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
            }
            continue;
        }
        uint64_t movesBB = 0;
        if (type == 1) movesBB = board.knightAttacks[from];
        else if (type == 2) movesBB = board.getBishopAttacks(from, board.occ);
        else if (type == 3) movesBB = board.getRookAttacks(from, board.occ);
        else if (type == 4) movesBB = board.getBishopAttacks(from, board.occ) | board.getRookAttacks(from, board.occ);
        else if (type == 5) movesBB = board.kingAttacks[from];
        movesBB &= oppPieces;
        while (movesBB) {
            int to = __builtin_ctzll(movesBB); movesBB &= movesBB - 1;
            out[count++] = Move{from, to, piece, board.getPiece(to), 0, false, -1, -1, 0, false, -1, board.enPassantTarget, board.castleWK, board.castleWQ, board.castleBK, board.castleBQ};
        }
    }
}

void Lawliet::doMove(Board& board, Move& m, uint64_t& hash, SearchContext& ctx) {
    if (ctx.hashStackIdx < 4096) ctx.hashStack[ctx.hashStackIdx++] = hash;

    hash ^= zobristSide;
    hash ^= zobristCastle[encodeCastling(board)];
    if (board.enPassantTarget >= 0) hash ^= zobristEp[board.enPassantTarget];

    int pIdx = pieceToZobristIndex(m.pieceMoved);
    hash ^= zobristPiece[m.fromSquare][pIdx];

    if (m.pieceCaptured != 0) {
        int capSq = m.wasEnPassant ? m.enPassantCapturedSquare : m.toSquare;
        int capIdx = pieceToZobristIndex(m.pieceCaptured);
        hash ^= zobristPiece[capSq][capIdx];
    }

    if (m.wasCastling) {
        int rookIdx = pieceToZobristIndex(m.rookPiece);
        hash ^= zobristPiece[m.rookFrom][rookIdx];
        hash ^= zobristPiece[m.rookTo][rookIdx];
    }

    int destIdx = m.promotionPiece != 0 ? pieceToZobristIndex(m.promotionPiece) : pIdx;
    hash ^= zobristPiece[m.toSquare][destIdx];

    board.applyMove(m);
    board.updateCastlingRightsFromMove(m.fromSquare, m.toSquare, m.wasEnPassant ? m.enPassantCapturedSquare : (m.pieceCaptured != 0 ? m.toSquare : -1));
    board.enPassantTarget = -1;
    if (std::abs(m.pieceMoved) == 1 && std::abs((m.toSquare / 8) - (m.fromSquare / 8)) == 2) {
        board.enPassantTarget = (m.fromSquare + m.toSquare) / 2;
    }
    board.turn = -board.turn;

    hash ^= zobristCastle[encodeCastling(board)];
    if (board.enPassantTarget >= 0) hash ^= zobristEp[board.enPassantTarget];
}

void Lawliet::undoMove(Board& board, Move& m, uint64_t& hash, SearchContext& ctx) {
    board.turn = -board.turn; board.revertMove(m);
    if (ctx.hashStackIdx > 0) hash = ctx.hashStack[--ctx.hashStackIdx];
}

void Lawliet::storeTT(uint64_t key, int depth, int score, TTFlag flag, const Move& bestMove, int ply) {
    if (activeTm && activeTm->shouldStop()) return;
    TTEntry& entry = transpositionTable[key & (TT_SIZE - 1)];

    uint64_t currentVal = entry.data.load(std::memory_order_relaxed);
    int currentDepth = 0;
    uint8_t currentAge = 0;
    if (currentVal != 0) {
        uint16_t currentKeySig = static_cast<uint16_t>((currentVal >> 48) & 0xFFFFULL);
        uint16_t keySig = static_cast<uint16_t>((key >> 48) & 0xFFFFULL);
        if (currentKeySig == keySig) {
            currentDepth = static_cast<int>((currentVal >> 16) & 0xFF);
            currentAge = static_cast<uint8_t>((currentVal >> 42) & 0x3F);
        }
    }

    uint16_t keySig = static_cast<uint16_t>((key >> 48) & 0xFFFFULL);
    // Eviction logic: replace if empty, or if new depth is greater, or if entry is old (aging)
    if (currentVal == 0 || depth >= currentDepth || currentAge != ttAge) {
        uint16_t fromSq = (bestMove.fromSquare != bestMove.toSquare) ? bestMove.fromSquare : 0xFFFF;
        uint16_t toSq = (bestMove.fromSquare != bestMove.toSquare) ? bestMove.toSquare : 0xFFFF;
        int16_t promo = bestMove.promotionPiece;

        uint64_t newVal = packData(scoreToTT(score, ply), depth, flag, fromSq, toSq, promo, ttAge, keySig);
        entry.data.store(newVal, std::memory_order_relaxed);
    }
}

bool Lawliet::probeTT(uint64_t key, int depth, int alpha, int beta, int& scoreOut, Move& bestMoveOut, int ply, SearchContext& ctx) {
    ctx.ttLookups++;
    const TTEntry& entry = transpositionTable[key & (TT_SIZE - 1)];
    uint64_t val = entry.data.load(std::memory_order_relaxed);
    if (val == 0) return false;

    uint16_t keySig = 0;
    int ttScore = 0;
    int ttDepth = 0;
    uint8_t ttFlag = 0;
    uint16_t fromSq = 0;
    uint16_t toSq = 0;
    int16_t promo = 0;
    uint8_t dummyAge = 0;
    unpackData(val, ttScore, ttDepth, ttFlag, fromSq, toSq, promo, dummyAge, keySig);

    uint16_t expectedSig = static_cast<uint16_t>((key >> 48) & 0xFFFFULL);
    if (keySig != expectedSig) return false;

    ctx.ttHits++;
    bestMoveOut = Move{};
    if (fromSq != toSq && fromSq < 64 && toSq < 64) {
        bestMoveOut.fromSquare = fromSq;
        bestMoveOut.toSquare = toSq;
        bestMoveOut.promotionPiece = promo;
    } else {
        bestMoveOut.fromSquare = -1;
        bestMoveOut.toSquare = -1;
    }

    if (ttDepth < depth) return false;
    int score = scoreFromTT(ttScore, ply);
    if (ttFlag == TT_EXACT) { scoreOut = score; return true; }
    if (ttFlag == TT_ALPHA && score <= alpha) { scoreOut = score; return true; }
    if (ttFlag == TT_BETA && score >= beta) { scoreOut = score; return true; }
    return false;
}

int Lawliet::scoreMove(const Move& m, const Board& board, int ply, const Move& ttMove, const SearchContext& ctx) const {
    if (ttMove.fromSquare != ttMove.toSquare && m.fromSquare == ttMove.fromSquare && m.toSquare == ttMove.toSquare && m.promotionPiece == ttMove.promotionPiece) return 20000000;
    if (m.pieceCaptured != 0 || m.wasEnPassant) {
        int seeScore = see(board, m.fromSquare, m.toSquare);
        if (seeScore >= 0) return 10000000 + mvvLva[board.getPieceType(m.pieceMoved)][board.getPieceType(m.pieceCaptured) + 1];
        else return 5000000 + seeScore;
    }
    if (ply < MAX_PLY) {
        if (m.fromSquare == ctx.killerMoves[ply][0].fromSquare && m.toSquare == ctx.killerMoves[ply][0].toSquare) return 8000000;

        // Countermove Heuristic ordering (placed optimally between Killer 1 and Killer 2)
        if (ply > 0 && !board.moveHistory.empty()) {
            Move prevMove = board.moveHistory.back();
            Move counterMove = ctx.counterMoveTable[prevMove.fromSquare][prevMove.toSquare];
            if (m.fromSquare == counterMove.fromSquare && m.toSquare == counterMove.toSquare && m.promotionPiece == counterMove.promotionPiece) {
                return 7500000;
            }
        }

        if (m.fromSquare == ctx.killerMoves[ply][1].fromSquare && m.toSquare == ctx.killerMoves[ply][1].toSquare) return 7000000;
    }

    // Sort quiet moves leveraging both local History and Continuation History tables
    int colorIdx = (board.turn == Board::WHITE) ? 0 : 1;
    int pieceType = board.getPieceType(m.pieceMoved);
    int score = ctx.historyTable[colorIdx][pieceType][m.toSquare];
    if (ply > 0 && !board.moveHistory.empty()) {
        Move prevMove = board.moveHistory.back();
        int prevPieceIdx = pieceToZobristIndex(prevMove.pieceMoved);
        if (prevPieceIdx >= 0 && prevPieceIdx < 12) {
            score += ctx.continuationHistory[prevPieceIdx][m.toSquare];
        }
    }
    return score;
}

void Lawliet::orderMoves(Move* moves, int count, const Board& board, int ply, const Move& ttMove, const SearchContext& ctx) const {
    std::sort(moves, moves + count, [&](const Move& a, const Move& b) { return scoreMove(a, board, ply, ttMove, ctx) > scoreMove(b, board, ply, ttMove, ctx); });
}

int Lawliet::quiescence(Board& board, int alpha, int beta, int ply, uint64_t hash, TimeManager& tm, SearchContext& ctx) {
    if (tm.stopFlag.load(std::memory_order_relaxed)) return 0;
    ctx.localNodes++;
    if ((ctx.localNodes & 2047) == 0) {
        tm.nodes += 2048;
        if (tm.shouldStop()) return 0;
    }
    if (ply >= MAX_PLY - 1) return evaluateBoard(board, alpha, beta, &ctx);

    bool inCheck = board.isInCheck(board.turn);

    // Stand pat (Static Evaluation) if not in check
    if (!inCheck) {
        int staticEval = evaluateBoard(board, alpha, beta, &ctx);
        if (staticEval >= beta) return beta;
        if (staticEval > alpha) alpha = staticEval;
    }

    if (ply > ctx.maxQuiescencePly) ctx.maxQuiescencePly = ply;

    // Generate either all legal evasions if in check or only captures if out of check
    if (inCheck) {
        generateLegalMoves(board, board.turn, ctx.moveBuffers[ply], ctx.moveCounts[ply]);
    } else {
        generateCaptures(board, board.turn, ctx.moveBuffers[ply], ctx.moveCounts[ply]);
    }

    const Move noMove{};
    orderMoves(ctx.moveBuffers[ply], ctx.moveCounts[ply], board, ply, noMove, ctx);

    int legalMovesSearched = 0;

    for (int i = 0; i < ctx.moveCounts[ply]; ++i) {
        if (tm.shouldStop()) return 0;
        Move m = ctx.moveBuffers[ply][i];
        if (!inCheck && see(board, m.fromSquare, m.toSquare) < 0) continue;

        doMove(board, m, hash, ctx);

        // Fast Legality Check
        if (board.isInCheck(-board.turn)) {
            undoMove(board, m, hash, ctx);
            continue;
        }
        legalMovesSearched++;

        int score = -quiescence(board, -beta, -alpha, ply + 1, hash, tm, ctx);
        undoMove(board, m, hash, ctx);

        if (tm.shouldStop()) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    // Q-Search mate handling
    if (inCheck && legalMovesSearched == 0) {
        return -INF + ply;
    }

    return alpha;
}

int Lawliet::negamax(Board& board, int depth, int alpha, int beta, int ply, uint64_t hash, TimeManager& tm, SearchContext& ctx, Move excludedMove) {
    // Check stop flag instantly on every node to block score leakage upon time out
    if (tm.stopFlag.load(std::memory_order_relaxed)) return 0;

    ctx.localNodes++;
    if ((ctx.localNodes & 2047) == 0) {
        tm.nodes += 2048;
        if (tm.shouldStop()) return 0;
    }
    if (ply >= MAX_PLY - 1) return evaluateBoard(board, alpha, beta, &ctx);

    // Fast Bounded Threefold Repetition Detection (NPS Optimizer)
    int lastIrreversible = 0;
    for (int i = static_cast<int>(board.moveHistory.size()) - 1; i >= 0; --i) {
        if (std::abs(board.moveHistory[i].pieceMoved) == 1 || board.moveHistory[i].pieceCaptured != 0) {
            lastIrreversible = i + 1;
            break;
        }
    }
    int scanStart = ctx.hashStackIdx - 1;
    int scanEnd = ctx.hashStackIdx - static_cast<int>(board.moveHistory.size() - lastIrreversible);
    scanEnd = std::max(0, scanEnd);
    for (int i = scanStart; i >= scanEnd; --i) {
        if (ctx.hashStack[i] == hash) {
            return 0; // Immediate draw return
        }
    }

    if (depth <= 0) return quiescence(board, alpha, beta, ply, hash, tm, ctx);

    int ttScore = 0; Move ttMove; ttMove.fromSquare = -1;
    bool hasTT = false;
    // Exempt excludedMove (Singular Search) from normal TT probes
    if (excludedMove.fromSquare == excludedMove.toSquare) {
        hasTT = probeTT(hash, depth, alpha, beta, ttScore, ttMove, ply, ctx);
        if (hasTT && ply > 0) return ttScore;
    }

    bool inCheck = board.isInCheck(board.turn);
    int staticEval = evaluateBoard(board, alpha, beta, &ctx);

    // Razoring
    if (depth == 1 && !inCheck && staticEval + 300 < alpha) {
        int qScore = quiescence(board, alpha - 300, beta, ply, hash, tm, ctx);
        if (qScore < alpha - 300) return qScore;
    }

    // Singular Extension Logic (PV-nodes, deep branches with valid TT moves)
    int extension = 0;
    if (depth >= 8 && hasTT && ttMove.fromSquare != -1 && !inCheck && excludedMove.fromSquare == excludedMove.toSquare) {
        int ttDepth = 0, dummyScore = 0; int16_t dummyPromo = 0; uint8_t ttFlag = 0, dummyAge = 0; uint16_t fromSq = 0, toSq = 0;
        TTEntry& entry = transpositionTable[hash & (TT_SIZE - 1)];
        uint64_t data = entry.data.load(std::memory_order_relaxed);
        uint16_t keySig = 0;
        unpackData(data, dummyScore, ttDepth, ttFlag, fromSq, toSq, dummyPromo, dummyAge, keySig);

        uint16_t expectedSig = static_cast<uint16_t>((hash >> 48) & 0xFFFFULL);
        if (keySig == expectedSig && ttDepth >= depth - 3 && (ttFlag == TT_EXACT || ttFlag == TT_BETA) && std::abs(ttScore) < INF - 1000) {
            int singularBeta = ttScore - 2 * depth; // Singular margin = 2 * depth
            int rDepth = depth - 3;

            // Search excluding the expected singular PV move
            int rScore = negamax(board, rDepth, singularBeta - 1, singularBeta, ply, hash, tm, ctx, ttMove);
            if (rScore < singularBeta) {
                extension = 1; // Singular move confirmed! Extend 1 ply.
            }
        }
    }

    // Internal Iterative Deepening (IID)
    if (depth >= 5 && ttMove.fromSquare == -1 && !inCheck) {
        int iidDepth = depth - 2;
        negamax(board, iidDepth, alpha, beta, ply, hash, tm, ctx, excludedMove);
        probeTT(hash, iidDepth, alpha, beta, ttScore, ttMove, ply, ctx);
    }

    // Reverse Futility Pruning (RFP) up to depth 6
    if (depth <= 6 && !inCheck && std::abs(beta) < INF - 1000) {
        int margin = 120 * depth;
        if (staticEval - margin >= beta) return staticEval - margin;
    }

    bool hasNonPawnMaterial = false;
    int turnColorIdx = (board.turn == Board::WHITE) ? 0 : 1;
    uint64_t nonPawns = board.colorBB[turnColorIdx] ^ board.pieceBB[turnColorIdx * 6];
    if (nonPawns ^ board.pieceBB[turnColorIdx * 6 + 5]) {
        hasNonPawnMaterial = true;
    }

    // Optimization 4: Depth-Scaled Null Move Pruning (NMP) Reduction
    if (depth >= 3 && !inCheck && ply > 0 && hasNonPawnMaterial) {
        uint64_t nullHash = hash ^ zobristSide; if (board.enPassantTarget != -1) nullHash ^= zobristEp[board.enPassantTarget];
        uint64_t epBackup = board.enPassantTarget; board.enPassantTarget = -1; board.turn = -board.turn;

        int R = 3 + depth / 6;
        int nullScore = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, nullHash, tm, ctx, excludedMove);

        board.turn = -board.turn; board.enPassantTarget = epBackup;
        if (nullScore >= beta) {
            if (nullScore >= INF - 1000) return beta;
            return nullScore;
        }
    }

    // --- ProbCut (Probability Cutoff) ---
    if (depth >= 5 && !inCheck && ply > 0 && std::abs(beta) < INF - 1000) {
        int rDepth = depth - 3;
        int probCutBeta = beta + 200; // Standard 200 cp margin

        // We run a fast shallow search with the elevated beta
        int probScore = negamax(board, rDepth, probCutBeta - 1, probCutBeta, ply + 1, hash, tm, ctx, excludedMove);
        if (probScore >= probCutBeta) {
            return beta; // Cutoff confirmed with high probability!
        }
    }

    bool futility = false;
    if (depth <= 2 && !inCheck && std::abs(alpha) < INF - 1000) { if (staticEval + depth * 150 <= alpha) futility = true; }

    generateLegalMoves(board, board.turn, ctx.moveBuffers[ply], ctx.moveCounts[ply]);
    if (ctx.moveCounts[ply] == 0) return inCheck ? -INF + ply : 0;

    // Check and One-Reply Extensions
    if (inCheck && ply < 16) {
        if (ctx.moveCounts[ply] == 1) {
            depth += 2; // Forced reply extension!
        } else {
            depth++;
        }
    }

    orderMoves(ctx.moveBuffers[ply], ctx.moveCounts[ply], board, ply, ttMove, ctx);

    // Safe Late Move Pruning
    int maxMoves = 3 + depth * depth;

    int bestScore = -INF; Move bestMove{}; TTFlag flag = TT_ALPHA;
    int movesSearched = 0;      // Tracks moves sent to negamax evaluation
    int legalMovesSearched = 0; // NEW: Tracks truly legal moves to detect mate/stalemate

    for (int i = 0; i < ctx.moveCounts[ply]; ++i) {
        if (tm.shouldStop()) return 0;
        Move m = ctx.moveBuffers[ply][i];

        if (excludedMove.fromSquare != -1 && m == excludedMove) continue;
        bool isQuiet = (m.pieceCaptured == 0 && m.promotionPiece == 0 && !m.wasCastling);

        // --- Static Exchange Evaluation (SEE) Pruning for Bad Captures ---
        if (depth <= 4 && !inCheck && (m.pieceCaptured != 0 || m.wasEnPassant)) {
            int seeScore = see(board, m.fromSquare, m.toSquare);
            if (seeScore < -17 * depth * depth) {
                undoMove(board, m, hash, ctx);
                continue;
            }
        }

        doMove(board, m, hash, ctx);

        // NEW: Fast legality check post-move.
        // Because turn is flipped in doMove, we check if the side that just moved is in check
        if (board.isInCheck(-board.turn)) {
            undoMove(board, m, hash, ctx);
            continue; // Skip this illegal pseudo-legal move
        }

        legalMovesSearched++; // Confirmed valid move
        bool givesCheck = board.isInCheck(board.turn);

        // Safe Late Move Pruning and Futility Pruning after doMove
        // Note: Using movesSearched here since pruning choices rely on position in the ordered queue
        if (depth <= 4 && !inCheck && isQuiet && !givesCheck && movesSearched >= maxMoves) {
            undoMove(board, m, hash, ctx);
            continue;
        }
        if (futility && isQuiet && !givesCheck && movesSearched > 0) {
            undoMove(board, m, hash, ctx);
            continue;
        }

        // Passed pawn moves are highly active and are exempted from LMR
        bool isPassedPawnMove = false;
        if (std::abs(m.pieceMoved) == 1) {
            if (m.pieceMoved > 0) {
                if (!(board.pieceBB[6] & pawnPassedMask[0][m.toSquare])) isPassedPawnMove = true;
            } else {
                if (!(board.pieceBB[0] & pawnPassedMask[1][m.toSquare])) isPassedPawnMove = true;
            }
        }

        int score;
        int nextDepth = depth - 1;
        if (m == ttMove && extension > 0) {
            nextDepth++; // Apply Singular Extension to PV move search
        }

        if (movesSearched == 0) {
            score = -negamax(board, nextDepth, -beta, -alpha, ply + 1, hash, tm, ctx);
        } else {
            // Optimization 3 & 7: Logarithmic & History-Based Late Move Reductions (LMR)
            int red = 0;
            bool isKiller = (ply < MAX_PLY) &&
            ((m.fromSquare == ctx.killerMoves[ply][0].fromSquare && m.toSquare == ctx.killerMoves[ply][0].toSquare) ||
            (m.fromSquare == ctx.killerMoves[ply][1].fromSquare && m.toSquare == ctx.killerMoves[ply][1].toSquare));

            if (nextDepth >= 3 && !inCheck && movesSearched >= 4 && isQuiet && !isPassedPawnMove && !isKiller) {
                if (!givesCheck) {
                    int dClamped = std::min(nextDepth, 127);
                    int mClamped = std::min(movesSearched, 255);
                    red = lmrTable[dClamped][mClamped];

                    // Helper thread diversification:
                    if (ctx.threadId > 0) {
                        if ((ctx.threadId + nextDepth + movesSearched) % 2 == 0) {
                            red++;
                        } else {
                            red = std::max(0, red - 1);
                        }
                    }

                    // History adjustment (Option 7)
                    int colorIdx = (board.turn == Board::WHITE) ? 0 : 1;
                    int pieceType = board.getPieceType(m.pieceMoved);
                    int hist = ctx.historyTable[colorIdx][pieceType][m.toSquare];
                    if (hist > 15000) red--;
                    else if (hist < -5000) red++;

                    red = std::max(0, std::min(red, nextDepth - 2));
                }
            }

            score = -negamax(board, nextDepth - red, -alpha - 1, -alpha, ply + 1, hash, tm, ctx);
            if (score > alpha) score = -negamax(board, nextDepth, -beta, -alpha, ply + 1, hash, tm, ctx);
        }

        undoMove(board, m, hash, ctx);
        if (tm.shouldStop()) return 0;

        movesSearched++; // Increment after a full evaluation has run

        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
            if (ply == 0) ctx.rootBestMove = m;
        }
        if (score > alpha) {
            alpha = score;
            flag = TT_EXACT;
            if (ply == 0) ctx.rootBestMove = m;
        }
        if (alpha >= beta) {
            flag = TT_BETA;
            ctx.failHighs++;
            if (movesSearched == 1) {
                ctx.fhf++;
            }
            // Store Heuristics ONLY on a genuine beta cutoff (refutation)
            if (isQuiet && ply < MAX_PLY) {
                ctx.killerMoves[ply][1] = ctx.killerMoves[ply][0];
                ctx.killerMoves[ply][0] = m;
                int colorIdx = (board.turn == Board::WHITE) ? 0 : 1;
                int pieceType = board.getPieceType(m.pieceMoved);
                ctx.historyTable[colorIdx][pieceType][m.toSquare] += std::min(2000, depth * depth);
                ctx.historyTable[colorIdx][pieceType][m.toSquare] = std::clamp(ctx.historyTable[colorIdx][pieceType][m.toSquare], -40000, 40000);

                // Countermove Heuristic update
                if (ply > 0 && !board.moveHistory.empty()) {
                    Move prevMove = board.moveHistory.back();
                    ctx.counterMoveTable[prevMove.fromSquare][prevMove.toSquare] = m;
                }

                // Continuation History update
                if (ply > 0 && !board.moveHistory.empty()) {
                    Move prevMove = board.moveHistory.back();
                    int prevPieceIdx = pieceToZobristIndex(prevMove.pieceMoved);
                    if (prevPieceIdx >= 0 && prevPieceIdx < 12) {
                        ctx.continuationHistory[prevPieceIdx][m.toSquare] += std::min(2000, depth * depth);
                        ctx.continuationHistory[prevPieceIdx][m.toSquare] = std::clamp(ctx.continuationHistory[prevPieceIdx][m.toSquare], -40000, 40000);
                    }
                }

                // History penalty to previous quiet moves
                for (int j = 0; j < i; ++j) {
                    Move prevMove = ctx.moveBuffers[ply][j];
                    bool prevQuiet = (prevMove.pieceCaptured == 0 && prevMove.promotionPiece == 0 && !prevMove.wasCastling);
                    if (prevQuiet) {
                        int pColorIdx = (board.turn == Board::WHITE) ? 0 : 1;
                        int pPieceType = board.getPieceType(prevMove.pieceMoved);
                        ctx.historyTable[pColorIdx][pPieceType][prevMove.toSquare] -= std::min(2000, depth * depth);
                        ctx.historyTable[pColorIdx][pPieceType][prevMove.toSquare] = std::clamp(ctx.historyTable[pColorIdx][pPieceType][prevMove.toSquare], -40000, 40000);

                        if (ply > 0 && !board.moveHistory.empty()) {
                            Move oppPrevMove = board.moveHistory.back();
                            int prevPieceIdx = pieceToZobristIndex(oppPrevMove.pieceMoved);
                            if (prevPieceIdx >= 0 && prevPieceIdx < 12) {
                                ctx.continuationHistory[prevPieceIdx][prevMove.toSquare] -= std::min(2000, depth * depth);
                                ctx.continuationHistory[prevPieceIdx][prevMove.toSquare] = std::clamp(ctx.continuationHistory[prevPieceIdx][prevMove.toSquare], -40000, 40000);
                            }
                        }
                    }
                }
            }
            break; // Beta cutoff triggered, stop searching remaining sibling nodes
        }
    }

    // NEW: Proper end of node verification
    // If absolutely zero pseudo-legal moves passed our post-move legality check,
    // the position is terminal. Check for checkmate vs. stalemate.
    if (legalMovesSearched == 0) {
        return inCheck ? (-INF + ply) : 0;
    }

    // Static Evaluation Correction History (CorrHist) update
    if (!inCheck && (bestMove.fromSquare == -1 || bestMove.pieceCaptured == 0) && excludedMove.fromSquare == -1) {
        int diff = bestScore - staticEval;
        diff = std::clamp(diff, -1000, 1000);

        uint64_t pKey = 0;
        uint64_t wp = board.pieceBB[0];
        while (wp) {
            pKey ^= zobristPiece[__builtin_ctzll(wp)][0];
            wp &= wp - 1;
        }
        uint64_t bp = board.pieceBB[6];
        while (bp) {
            pKey ^= zobristPiece[__builtin_ctzll(bp)][6];
            bp &= bp - 1;
        }

        int sideIdx = (board.turn == Board::WHITE) ? 0 : 1;
        int& entry = ctx.corrHist[sideIdx][pKey % 16384];
        int weight = std::min(128, depth * depth);
        entry = (entry * (1024 - weight) + diff * weight) / 1024;
        entry = std::clamp(entry, -8192, 8192);
    }

    storeTT(hash, depth, bestScore, flag, bestMove, ply);
    return bestScore;
}

std::string Lawliet::squareToUci(int sq) { return std::string(1, 'a' + (sq % 8)) + std::string(1, '1' + (7 - sq / 8)); }

std::string Lawliet::extractPv(Board& board, uint64_t hash) {
    std::string pv = ""; Board temp = board; uint64_t h = hash;
    for (int i = 0; i < 20; ++i) {
        TTEntry& entry = transpositionTable[h & (TT_SIZE - 1)];
        uint64_t data = entry.data.load(std::memory_order_relaxed);
        if (data == 0) break;
        int score = 0, depth = 0; uint8_t flag = 0; uint16_t fromSq = 0, toSq = 0; int16_t promo = 0; uint8_t dummyAge = 0; uint16_t keySig = 0;
        unpackData(data, score, depth, flag, fromSq, toSq, promo, dummyAge, keySig);
        uint16_t expectedSig = static_cast<uint16_t>((h >> 48) & 0xFFFFULL);
        if (keySig != expectedSig) break;
        if (fromSq == toSq || fromSq >= 64 || toSq >= 64) break;
        Move m; m.fromSquare = fromSq; m.toSquare = toSq; m.promotionPiece = promo;
        std::string moveStr = squareToUci(m.fromSquare) + squareToUci(m.toSquare);
        if (m.promotionPiece != 0) {
            int pType = std::abs(m.promotionPiece);
            if (pType == 5) moveStr += "q";
            else if (pType == 4) moveStr += "r";
            else if (pType == 3) moveStr += "b";
            else if (pType == 2) moveStr += "n";
        }
        pv += moveStr + " ";
        if (!temp.makeMove(m.fromSquare, m.toSquare, std::abs(m.promotionPiece))) break;
        h = computeHash(temp);
    }
    return pv;
}

uint64_t OpeningBook::computePolyglotHash(const Board& board) const {
    initPolyglotKeys(); uint64_t hash = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int piece = board.getPiece(sq); if (piece == 0) continue;
        int pType = std::abs(piece) - 1, pColor = (piece > 0) ? 1 : 0, polyIdx = pType * 2 + pColor;
        hash ^= polyglotPiece[polyIdx][(7 - (sq / 8)) * 8 + (sq % 8)];
    }
    if (board.castleWK) hash ^= polyglotCastle[0]; if (board.castleWQ) hash ^= polyglotCastle[1]; if (board.castleBK) hash ^= polyglotCastle[2]; if (board.castleBQ) hash ^= polyglotCastle[3];
    if (board.enPassantTarget != -1) {
        bool epPossible = false; int colorIdx = (board.turn == Board::WHITE) ? 0 : 1; uint64_t myPawns = board.pieceBB[board.turn == Board::WHITE ? 0 : 6];
        while (myPawns) { if (board.pawnAttacks[colorIdx][__builtin_ctzll(myPawns)] & (1ULL << board.enPassantTarget)) { epPossible = true; break; } myPawns &= myPawns - 1; }
        if (epPossible) hash ^= polyglotEp[board.enPassantTarget % 8];
    }
    if (board.turn == Board::WHITE) hash ^= polyglotSide;
    return hash;
}

uint16_t OpeningBook::lookup(uint64_t polyKey) const {
    auto it = std::lower_bound(entries.begin(), entries.end(), polyKey, [](const BookEntry& e, uint64_t key) { return e.key < key; });
    if (it == entries.end() || it->key != polyKey) return 0;
    std::vector<BookEntry> candidates; uint32_t totalWeight = 0;
    while (it != entries.end() && it->key == polyKey) { if (it->weight > 0) { candidates.push_back(*it); totalWeight += it->weight; } ++it; }
    if (candidates.empty()) return 0;
    std::random_device rd; std::mt19937 gen(rd()); std::uniform_int_distribution<uint32_t> dis(0, totalWeight - 1);
    uint32_t choice = dis(gen), currentWeight = 0;
    for (const auto& entry : candidates) { currentWeight += entry.weight; if (choice < currentWeight) return entry.move; }
    return candidates[0].move;
}

Lawliet::Lawliet(int depth) : maxDepth(depth) {
    initTables(); transpositionTable.resize(TT_SIZE);
    pawnTable.resize(PAWN_SIZE); // Initializing Pawn Hash Table (PHT)
    std::mt19937_64 rng(0x4C41776C696574ULL);
    for (int sq = 0; sq < 64; ++sq) for (int p = 0; p < 12; ++p) zobristPiece[sq][p] = rng();
    for (int c = 0; c < 16; ++c) zobristCastle[c] = rng();
    for (int e = 0; e < 64; ++e) zobristEp[e] = rng();
    zobristSide = rng(); book.load("book.bin");
}

void Lawliet::loadBook(const std::string& path) { if (!book.load(path)) std::cout << "info string Warning: Opening book could not be loaded from path: " << path << std::endl; }
void Lawliet::searchWorker(Board board, TimeManager& tm, int threadId, Move& outBestMove) { auto ctx = std::make_unique<SearchContext>(); outBestMove = thinkThread(board, tm, *ctx, threadId); }
Move Lawliet::think(Board& board) { TimeManager tm; tm.startInfiniteSearch(); return think(board, tm); }

Move Lawliet::think(Board& board, TimeManager& tm) {
    activeTm = &tm; int count = 0; Move moves[256]; generateLegalMoves(board, board.turn, moves, count);
    if (count == 0) { activeTm = nullptr; return Move{}; }
    if (count == 1) { std::cout << "info string Only one legal move. Playing instantly." << std::endl; activeTm = nullptr; return moves[0]; }
    if (book.isOpen()) {
        uint64_t polyKey = book.computePolyglotHash(board); uint16_t bookMoveRaw = book.lookup(polyKey);
        if (bookMoveRaw != 0) {
            int polyFrom = (bookMoveRaw >> 6) & 0x3F, polyTo = bookMoveRaw & 0x3F;
            int from = (7 - (polyFrom / 8)) * 8 + (polyFrom % 8), to = (7 - (polyTo / 8)) * 8 + (polyTo % 8);
            int piece = board.getPiece(from);
            if (std::abs(piece) == 6) {
                if (from == 60 && to == 63) to = 62; else if (from == 60 && to == 56) to = 58;
                else if (from == 4 && to == 7) to = 6; else if (from == 4 && to == 0) to = 2;
            }
            int promoRaw = (bookMoveRaw >> 12) & 7, promo = (promoRaw == 1) ? 2 : (promoRaw == 2 ? 3 : (promoRaw == 3 ? 4 : (promoRaw == 4 ? 5 : 0)));
            for (int i = 0; i < count; ++i) {
                if (moves[i].fromSquare == from && moves[i].toSquare == to && promo == std::abs(moves[i].promotionPiece)) {
                    std::cout << "info string Playing opening book move: " << squareToUci(from) << squareToUci(to) << std::endl;
                    activeTm = nullptr; return moves[i];
                }
            }
        }
    }
    const int numThreads = 4; std::vector<std::thread> workers; std::vector<Move> threadBestMoves(numThreads);
    for (int i = 1; i < numThreads; ++i) workers.push_back(std::thread(&Lawliet::searchWorker, this, board, std::ref(tm), i, std::ref(threadBestMoves[i])));
    auto masterCtx = std::make_unique<SearchContext>(); Move bestMove = thinkThread(board, tm, *masterCtx, 0);

    // CRITICAL: We must stop the timer to signal all helper threads to exit immediately!
    tm.stop();

    for (auto& t : workers) { if (t.joinable()) t.join(); }

    int elapsed = static_cast<int>(tm.getElapsedMs());
    int64_t totalNodes = tm.nodes.load();
    printSearchStats(*masterCtx, maxDepth, masterCtx->bestScore, elapsed, totalNodes);

    activeTm = nullptr; return bestMove;
}

Move Lawliet::thinkThread(Board& board, TimeManager& tm, SearchContext& ctx, int threadId) {
    ctx.threadId = threadId; // Set threadId in context!
    {
        Board tempBoard = board;
        std::vector<uint64_t> historyHashes;
        while (!tempBoard.moveHistory.empty()) {
            tempBoard.undoMove();
            historyHashes.push_back(computeHash(tempBoard));
        }
        std::reverse(historyHashes.begin(), historyHashes.end());
        ctx.hashStackIdx = 0;
        for (uint64_t h : historyHashes) {
            if (ctx.hashStackIdx < 4096) {
                ctx.hashStack[ctx.hashStackIdx++] = h;
            }
        }
    }

    generateLegalMoves(board, board.turn, ctx.moveBuffers[0], ctx.moveCounts[0]);
    if (ctx.moveCounts[0] == 0) return Move{};

    // Find the first fully legal move as the fallback to prevent illegal move leaks
    Move bestMove{};
    for (int i = 0; i < ctx.moveCounts[0]; ++i) {
        if (board.leavesKingInCheck(ctx.moveBuffers[0][i].fromSquare, ctx.moveBuffers[0][i].toSquare, std::abs(ctx.moveBuffers[0][i].promotionPiece))) {
            bestMove = ctx.moveBuffers[0][i];
            break;
        }
    }
    Move completedDepthBestMove = bestMove;
    Move lastCompletedBestMove{};

    int bestMoveStability = 0; double complexityScore = 0.0;
    if (threadId == 0 && !tm.infinite.load() && tm.allocatedTimeMs.load() > 0) {
        int captures = 0, promos = 0, checks = 0;
        for (int i = 0; i < ctx.moveCounts[0]; ++i) {
            const Move& m = ctx.moveBuffers[0][i]; if (m.pieceCaptured != 0 || m.wasEnPassant) captures++; if (m.promotionPiece != 0) promos++;
            board.applyMove(const_cast<Move&>(m)); if (board.isInCheck(-board.turn)) checks++; board.revertMove(m);
        }
        int pieceCount = 0; for (int i = 0; i < 12; ++i) pieceCount += __builtin_popcountll(board.pieceBB[i]);
        if (board.isInCheck(board.turn)) complexityScore += 1.5;
        complexityScore += (captures * 0.15) + (promos * 0.5) + (checks * 0.2) + (ctx.moveCounts[0] * 0.02);
        int ourKingSq = board.findKing(board.turn);
        if (ourKingSq != -1) {
            uint64_t kArea = Board::kingAttacks[ourKingSq];
            while (kArea) { int sq = __builtin_ctzll(kArea); kArea &= kArea - 1; if (board.isSquareAttacked(sq, -board.turn)) complexityScore += 0.25; }
        }
        double compMult = std::max(0.7, std::min(2.2, 0.65 + (complexityScore * 0.22))), phaseMult = pieceCount > 24 ? 1.25 : (pieceCount < 12 ? 0.8 : 1.0);

        int baseTime = tm.allocatedTimeMs.load();
        int targetTime = static_cast<int>(baseTime * compMult * phaseMult);
        targetTime = static_cast<int>(baseTime * 0.6 + targetTime * 0.4);

        if (tm.totalTimeMs.load() > 0) { int maxAllowed = tm.totalTimeMs.load() / 5; if (targetTime > maxAllowed) targetTime = maxAllowed; }
        if (targetTime < 10) targetTime = 10;
        tm.allocatedTimeMs.store(targetTime);
    }
    int originalAllocatedTimeMs = tm.allocatedTimeMs.load(); uint64_t hash = computeHash(board);
    int lastScore = 0; const int effMaxDepth = maxDepth;

    // Fully verifies move legality against checks and pins prior to execution
    auto validateMove = [&](const Move& m) {
        if (m.fromSquare == m.toSquare) return false;
        bool found = false;
        for (int i = 0; i < ctx.moveCounts[0]; ++i) {
            if (ctx.moveBuffers[0][i].fromSquare == m.fromSquare &&
                ctx.moveBuffers[0][i].toSquare == m.toSquare &&
                ctx.moveBuffers[0][i].promotionPiece == m.promotionPiece) {
                found = true;
            break;
                }
        }
        if (!found) return false;
        return board.leavesKingInCheck(m.fromSquare, m.toSquare, std::abs(m.promotionPiece));
    };

    int startingDepth = threadId > 0 ? 1 + (threadId % 2) : 1;
    int prevScore = 0;

    // TT Age incrementation on master thread startup
    if (threadId == 0) {
        ttAge = (ttAge + 1) & 63;
    }

    for (int depth = startingDepth; depth <= effMaxDepth; ++depth) {
        if (threadId == 0) { if (tm.shouldStopAtRoot(depth)) break; } else { if (tm.shouldStop()) break; }
        ctx.rootBestMove = Move{};
        prevScore = lastScore;

        // Fixed Aspiration Bounds with Gradual Widening
        int alpha = -INF;
        int beta = INF;
        int window = 45;

        if (depth >= 5) {
            alpha = lastScore - window;
            beta = lastScore + window;
        }

        while (true) {
            int score = negamax(board, depth, alpha, beta, 0, hash, tm, ctx);
            if (tm.shouldStop()) break;

            if (score <= alpha) {
                alpha = std::max(-INF, alpha - window);
                window += window / 2;
                if (threadId == 0 && !tm.infinite.load() && tm.totalTimeMs.load() > 0) {
                    int maxAllowed = tm.totalTimeMs.load() / 3;
                    int newAlloc = static_cast<int>(tm.allocatedTimeMs.load() * 1.3);
                    if (newAlloc > maxAllowed) newAlloc = maxAllowed;
                    tm.allocatedTimeMs.store(newAlloc);
                }
            } else if (score >= beta) {
                beta = std::min(INF, beta + window);
                window += window / 2;
                if (threadId == 0 && !tm.infinite.load() && tm.totalTimeMs.load() > 0) {
                    int maxAllowed = tm.totalTimeMs.load() / 3;
                    int newAlloc = static_cast<int>(tm.allocatedTimeMs.load() * 1.3);
                    if (newAlloc > maxAllowed) newAlloc = maxAllowed;
                    tm.allocatedTimeMs.store(newAlloc);
                }
            } else {
                lastScore = score;
                break;
            }
        }

        if (tm.shouldStop()) break;
        if (depth > 1) { if (ctx.rootBestMove == lastCompletedBestMove) bestMoveStability++; else bestMoveStability = 0; }
        lastCompletedBestMove = ctx.rootBestMove;
        ctx.bestScore = lastScore;
        if (validateMove(ctx.rootBestMove)) completedDepthBestMove = ctx.rootBestMove;
        else {
            TTEntry& entry = transpositionTable[hash & (TT_SIZE - 1)];
            uint64_t data = entry.data.load(std::memory_order_relaxed);
            if (data != 0) {
                int scoreTT = 0, dTT = 0; uint8_t fTT = 0; uint16_t fromSq = 0, toSq = 0; int16_t promo = 0; uint8_t dummyAge = 0; uint16_t keySig = 0;
                unpackData(data, scoreTT, dTT, fTT, fromSq, toSq, promo, dummyAge, keySig);
                uint16_t expectedSig = static_cast<uint16_t>((hash >> 48) & 0xFFFFULL);
                if (keySig == expectedSig) {
                    Move eMove; eMove.fromSquare = fromSq; eMove.toSquare = toSq; eMove.promotionPiece = promo;
                    if (validateMove(eMove)) completedDepthBestMove = eMove;
                }
            }
        }
        if (threadId == 0 && tm.onInfo) tm.onInfo(depth, lastScore, tm.nodes.load(), (int)tm.getElapsedMs(), extractPv(board, hash));
        if (std::abs(lastScore) > INF - 1000) break;
        if (threadId == 0 && depth > 3 && !tm.infinite.load() && tm.totalTimeMs.load() > 0) {
            double instMult = std::abs(lastScore - prevScore) > 100 ? 1.5 : (std::abs(lastScore - prevScore) > 40 ? 1.2 : (std::abs(lastScore - prevScore) < 15 ? 0.85 : 1.0));
            double stabMult = bestMoveStability >= 3 ? 0.75 : (bestMoveStability == 0 ? 1.3 : 1.0);
            int nextAllocated = static_cast<int>(originalAllocatedTimeMs * instMult * stabMult);
            int maxAllowed = tm.totalTimeMs.load() / 4; if (nextAllocated > maxAllowed) nextAllocated = maxAllowed;
            if (nextAllocated < 10) nextAllocated = 10;
            tm.allocatedTimeMs.store(nextAllocated);
            if (depth >= 10 && bestMoveStability >= 5 && std::abs(lastScore - prevScore) < 8 && complexityScore < 1.5) break;
        }
    }
    return completedDepthBestMove;
}
