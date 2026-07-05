#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include "board.hpp"
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <cstring>

uint64_t Board::knightAttacks[64];
uint64_t Board::kingAttacks[64];
uint64_t Board::pawnAttacks[2][64];
uint64_t Board::rayMasks[64][8];
bool Board::tablesInitialized = false;

Board::Board() {
    // Explicitly zero-initialize raw arrays and primitives to prevent garbage evaluations
    std::memset(pieceBB, 0, sizeof(pieceBB));
    std::memset(colorBB, 0, sizeof(colorBB));
    occ = 0;
    turn = WHITE;
    enPassantTarget = -1;
    castleWK = castleWQ = castleBK = castleBQ = false;
    gameResult = 0;
    mgPst = 0;
    egPst = 0;

    initAttackTables();
    reset();
}

void Board::reset() {
    loadFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Board::loadFen(const std::string& fen) {
    // Completely clear board arrays and reset evaluation accumulators to 0
    std::memset(pieceBB, 0, sizeof(pieceBB));
    std::memset(colorBB, 0, sizeof(colorBB));
    occ = 0;
    mgPst = 0;
    egPst = 0;
    gameResult = 0;
    moveHistory.clear();
    moveNotation.clear();

    std::istringstream iss(fen);
    std::string boardPart;
    if (!(iss >> boardPart)) return false;

    // Parse piece positions
    int rank = 0, file = 0;
    for (char c : boardPart) {
        if (c == '/') {
            rank++;
            file = 0;
        } else if (c >= '0' && c <= '9') {
            file += (c - '0');
        } else {
            int sq = rank * 8 + file;
            int piece = 0;
            switch (c) {
                case 'P': piece = 1; break;
                case 'N': piece = 2; break;
                case 'B': piece = 3; break;
                case 'R': piece = 4; break;
                case 'Q': piece = 5; break;
                case 'K': piece = 6; break;
                case 'p': piece = -1; break;
                case 'n': piece = -2; break;
                case 'b': piece = -3; break;
                case 'r': piece = -4; break;
                case 'q': piece = -5; break;
                case 'k': piece = -6; break;
                default: return false;
            }
            setPiece(sq, piece);
            file++;
        }
    }

    // Parse active color
    std::string turnPart;
    if (iss >> turnPart) {
        turn = (turnPart == "w") ? WHITE : BLACK;
    } else {
        turn = WHITE;
    }

    // Parse castling rights
    std::string castlePart;
    castleWK = castleWQ = castleBK = castleBQ = false;
    if (iss >> castlePart) {
        if (castlePart != "-") {
            for (char c : castlePart) {
                if (c == 'K') castleWK = true;
                else if (c == 'Q') castleWQ = true;
                else if (c == 'k') castleBK = true;
                else if (c == 'q') castleBQ = true;
            }
        }
    }

    // Parse en-passant square
    std::string epPart;
    enPassantTarget = -1;
    if (iss >> epPart) {
        if (epPart != "-") {
            int f = epPart[0] - 'a';
            int r = '8' - epPart[1];
            enPassantTarget = r * 8 + f;
        }
    }

    return true;
}

void Board::initAttackTables() {
    if (tablesInitialized) return;
    tablesInitialized = true;

    for (int sq = 0; sq < 64; ++sq) {
        int r = sq / 8, f = sq % 8;

        uint64_t ray = 0;
        for (int curR = r - 1; curR >= 0; --curR) ray |= (1ULL << (curR * 8 + f));
        rayMasks[sq][0] = ray;

        ray = 0;
        for (int curR = r + 1; curR < 8; ++curR) ray |= (1ULL << (curR * 8 + f));
        rayMasks[sq][1] = ray;

        ray = 0;
        for (int curF = f + 1; curF < 8; ++curF) ray |= (1ULL << (r * 8 + curF));
        rayMasks[sq][2] = ray;

        ray = 0;
        for (int curF = f - 1; curF >= 0; --curF) ray |= (1ULL << (r * 8 + curF));
        rayMasks[sq][3] = ray;

        ray = 0;
        for (int i = 1; r - i >= 0 && f + i < 8; ++i) ray |= (1ULL << ((r - i) * 8 + (f + i)));
        rayMasks[sq][4] = ray;

        ray = 0;
        for (int i = 1; r + i < 8 && f - i >= 0; ++i) ray |= (1ULL << ((r + i) * 8 + (f - i)));
        rayMasks[sq][5] = ray;

        ray = 0;
        for (int i = 1; r - i >= 0 && f - i >= 0; ++i) ray |= (1ULL << ((r - i) * 8 + (f - i)));
        rayMasks[sq][6] = ray;

        ray = 0;
        for (int i = 1; r + i < 8 && f + i < 8; ++i) ray |= (1ULL << ((r + i) * 8 + (f + i)));
        rayMasks[sq][7] = ray;
    }

    for (int sq = 0; sq < 64; ++sq) {
        int r = sq / 8, f = sq % 8;
        uint64_t bit = 1ULL << sq;

        uint64_t k = 0;
        if (r > 1 && f > 0) k |= (1ULL << (sq - 17));
        if (r > 1 && f < 7) k |= (1ULL << (sq - 15));
        if (r > 0 && f > 1) k |= (1ULL << (sq - 10));
        if (r > 0 && f < 6) k |= (1ULL << (sq - 6));
        if (r < 7 && f > 1) k |= (1ULL << (sq + 6));
        if (r < 7 && f < 6) k |= (1ULL << (sq + 10));
        if (r < 6 && f > 0) k |= (1ULL << (sq + 15));
        if (r < 6 && f < 7) k |= (1ULL << (sq + 17));
        knightAttacks[sq] = k;

        uint64_t ki = 0;
        if (r > 0) ki |= (bit >> 8);
        if (r < 7) ki |= (bit << 8);
        if (f > 0) ki |= (bit >> 1);
        if (f < 7) ki |= (bit << 1);
        if (r > 0 && f > 0) ki |= (bit >> 9);
        if (r > 0 && f < 7) ki |= (bit >> 7);
        if (r < 7 && f > 0) ki |= (bit << 7);
        if (r < 7 && f < 7) ki |= (bit << 9);
        kingAttacks[sq] = ki;

        if (r > 0 && f > 0) pawnAttacks[0][sq] |= (bit >> 9);
        if (r > 0 && f < 7) pawnAttacks[0][sq] |= (bit >> 7);
        if (r < 7 && f > 0) pawnAttacks[1][sq] |= (bit << 7);
        if (r < 7 && f < 7) pawnAttacks[1][sq] |= (bit << 9);
    }
}

uint64_t Board::getRookAttacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    uint64_t ray = rayMasks[sq][0];
    uint64_t blockers = ray & occ;
    if (blockers) {
        int blockerSq = 63 - __builtin_clzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][0]);
    } else attacks |= ray;

    ray = rayMasks[sq][1];
    blockers = ray & occ;
    if (blockers) {
        int blockerSq = __builtin_ctzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][1]);
    } else attacks |= ray;

    ray = rayMasks[sq][2];
    blockers = ray & occ;
    if (blockers) {
        int blockerSq = __builtin_ctzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][2]);
    } else attacks |= ray;

    ray = rayMasks[sq][3];
    blockers = ray & occ;
    if (blockers) {
        int blockerSq = 63 - __builtin_clzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][3]);
    } else attacks |= ray;

    return attacks;
}

uint64_t Board::getBishopAttacks(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    uint64_t ray = rayMasks[sq][4];
    uint64_t blockers = ray & occ;
    if (blockers) {
        int blockerSq = 63 - __builtin_clzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][4]);
    } else attacks |= ray;

    ray = rayMasks[sq][5];
    blockers = ray & occ;
    if (blockers) {
        int blockerSq = __builtin_ctzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][5]);
    } else attacks |= ray;

    ray = rayMasks[sq][6];
    blockers = ray & occ;
    if (blockers) {
        int blockerSq = 63 - __builtin_clzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][6]);
    } else attacks |= ray;

    ray = rayMasks[sq][7];
    blockers = occ & ray;
    if (blockers) {
        int blockerSq = __builtin_ctzll(blockers);
        attacks |= (ray & ~rayMasks[blockerSq][7]);
    } else attacks |= ray;

    return attacks;
}

const int Board::pawnTableMidgame[64] = {
    0,  0,  0,  0,  0,  0,  0,  0, 100,100, 92, 98, 98, 92,100,100,
    -7,  3, 26, 34, 34, 26,  3, -7, -37, -2,-11,  5,  5,-11, -2,-37,
    -43,-14,-20, -1, -1,-20,-14,-43, -36, -5,-17,-19,-19,-17, -5,-36,
    -45, -2,-18,-36,-36,-18, -2,-45, 0,  0,  0,  0,  0,  0,  0,  0
};

const int Board::knightTableMidgame[64] = {
    -100,-86,-80,  3,  3,-80,-86,-100, -67,-32, 55, 14, 14, 55,-32,-67,
    -13, 55, 49, 70, 70, 49, 55,-13, 4, 10, 32, 33, 33, 32, 10,  4,
    -18,  6, 10, 10, 10, 10,  6,-18, -23,  1,  9,  7,  7,  9,  1,-23,
    -29,-45, -9, -7, -7, -9,-45,-29, -68,-23,-46,-33,-33,-46,-23,-68
};

const int Board::bishopTableMidgame[64] = {
    -23, -2,-60,-51,-51,-60, -2,-23, -40, 15, 12,  9,  9, 12, 15,-40,
    -8, 41, 46, 42, 42, 46, 41, -8, 1,  6, 30, 47, 47, 30,  6,  1,
    -2, 13, 15, 30, 30, 15, 13, -2, 8, 19, 24, 17, 17, 24, 19,  8,
    9, 28, 18,  6,  6, 18, 28,  9, -28, -7, -9,-19,-19, -9, -7,-28
};

const int Board::rookTableMidgame[64] = {
    15, 20, 20, 25, 25, 20, 20, 15, 20, 25, 25, 30, 30, 25, 25, 20,
    5, 10, 15, 20, 20, 15, 10,  5, -10, -5,  5, 10, 10,  5, -5,-10,
    -20,-10,  0,  5,  5,  0,-10,-20, -25,-15, -5,  0,  0, -5,-15,-25,
    -30,-20,-10, -5, -5,-10,-20,-30, -15,-10,  0, 10, 10,  0,-10,-15
};

const int Board::queenTableMidgame[64] = {
    -10, -5, -5, -5, -5, -5, -5,-10,  -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,  -5,  0,  5, 10, 10,  5,  0, -5,
    -5,  0,  5, 10, 10,  5,  0, -5,  -5,  0,  5,  5,  5,  5,  0, -5,
    -10, -5,  5,  5,  5,  5, -5,-10, -10, -5, -5,  5,  5, -5, -5,-10
};

const int Board::kingTableMidgame[64] = {
    -5, 10, 10,-13,-13, 10, 10, -5, 0,  3, 10,  0,  0, 10,  3,  0,
    17, 10, 10,-12,-12, 10, 10, 17, -36, -4, -7,-26,-26, -7, -4,-36,
    -63,-27,-48,-85,-85,-48,-27,-63, -14, -4,-35,-57,-57,-35, -4,-14,
    43, 32,-15,-45,-45,-15, 32, 43, 47, 60,  1, 29, 29,  1, 60, 47
};

const int Board::pawnTableEndgame[64] = {
    0,  0,  0,  0,  0,  0,  0,  0, 130,130,128,117,117,128,130,130,
    77, 84, 56, 48, 48, 56, 84, 77, 18, 11,  0,-10,-10,  0, 11, 18,
    -2, -3,-14,-18,-18,-14, -3, -2, -10, -7,-15, -8, -8,-15, -7,-10,
    -4, -4, -2,  2,  2, -2, -4, -4, 0,  0,  0,  0,  0,  0,  0,  0
};

const int Board::knightTableEndgame[64] = {
    -100,-59,-32,-50,-50,-32,-59,-100, -48,-30,-46,-24,-24,-46,-30,-48,
    -51,-40,-15,-21,-21,-15,-40,-51, -37,-15, -3,  2,  2, -3,-15,-37,
    -38,-21, -3,  2,  2, -3,-21,-38, -44,-31,-24, -9, -9,-24,-31,-44,
    -62,-38,-34,-25,-25,-34,-38,-62, -65,-75,-38,-38,-38,-38,-75,-65
};

const int Board::bishopTableEndgame[64] = {
    -29,-29,-18,-13,-13,-18,-29,-29, -17,-17,-13,-19,-19,-13,-17,-17,
    -10,-20,-12,-17,-17,-12,-20,-10, -13, -5, -3, -4, -4, -3, -5,-13,
    -19,-13, -3,  0,  0, -3,-13,-19, -26,-18, -9, -2, -2, -9,-18,-26,
    -35,-32,-21,-10,-10,-21,-32,-35, -32,-22,-36,-17,-17,-36,-22,-32
};

const int Board::rookTableEndgame[64] = {
    8,  3, 15, 10, 10, 15,  3,  8, 4,  7,  2,  1,  1,  2,  7,  4,
    4,  0,  0,  3,  3,  0,  0,  4, 7,  3,  8,  0,  0,  8,  3,  7,
    1,  2,  5,  1,  1,  5,  2,  1, -3, -1, -6, -3, -3, -6, -1, -3,
    0, -7, -1, -1, -1, -1, -7,  0, -12,  5,  2, -1, -1,  2,  5,-12
};

const int Board::queenTableEndgame[64] = {
    -4,  8, 13, 18, 18, 13,  8, -4, -16, 29, 25, 49, 49, 25, 29,-16,
    -23,-10, 20, 50, 50, 20,-10,-23, 6, 42, 27, 52, 52, 27, 42,  6,
    -11, 28, 23, 43, 43, 23, 28,-11, -9,-27, 11,  3,  3, 11,-27, -9,
    -31,-37,-46,-19,-19,-46,-37,-31, -48,-38,-33,-50,-50,-33,-38,-48
};

const int Board::kingTableEndgame[64] = {
    -38,-18,-11,-22,-22,-11,-18,-38, -6, 11, 15,  6,  6, 15, 11, -6,
    -1, 24, 24,  9,  9, 24, 24, -1, -6, 15, 20, 20, 20, 20, 15, -6,
    -14,  0, 19, 29, 29, 19,  0,-14, -18, -4, 11, 21, 21, 11, -4,-18,
    -37,-21,  0,  9,  9,  0,-21,-37, -70,-47,-26,-42,-42,-26,-47,-70
};

const int Board::pieceValuesMidgame[6] = {100, 325, 330, 500, 975, 20000};
const int Board::pieceValuesEndgame[6] = {100, 320, 325, 510, 950, 20000};

const int* Board::pstMidgame[6] = {
    Board::pawnTableMidgame, Board::knightTableMidgame, Board::bishopTableMidgame,
    Board::rookTableMidgame, Board::queenTableMidgame, Board::kingTableMidgame
};
const int* Board::pstEndgame[6] = {
    Board::pawnTableEndgame, Board::knightTableEndgame, Board::bishopTableEndgame,
    Board::rookTableEndgame, Board::queenTableEndgame, Board::kingTableEndgame
};

bool Board::isColorMatch(int square, int color) const {
    uint64_t bit = 1ULL << square;
    return (color == WHITE) ? (colorBB[0] & bit) : (colorBB[1] & bit);
}

bool Board::isSquareAttacked(int square, int attackerColor) const {
    int colorIdx = (attackerColor == WHITE) ? 0 : 1;
    if (pawnAttacks[colorIdx ^ 1][square] & pieceBB[colorIdx * 6]) return true;
    if (knightAttacks[square] & pieceBB[colorIdx * 6 + 1]) return true;
    if (kingAttacks[square] & pieceBB[colorIdx * 6 + 5]) return true;

    uint64_t bishop = pieceBB[colorIdx * 6 + 2];
    uint64_t rook = pieceBB[colorIdx * 6 + 3];
    uint64_t queen = pieceBB[colorIdx * 6 + 4];

    if (getBishopAttacks(square, occ) & (bishop | queen)) return true;
    if (getRookAttacks(square, occ) & (rook | queen)) return true;
    return false;
}

bool Board::isInCheck(int color) const {
    int kingSq = findKing(color);
    return kingSq != -1 && isSquareAttacked(kingSq, -color);
}

int Board::findKing(int color) const {
    uint64_t kingBB = pieceBB[(color == WHITE) ? 5 : 11];
    return kingBB ? __builtin_ctzll(kingBB) : -1;
}

bool Board::isPawnPromotion(int from, int to) const {
    int piece = getPiece(from);
    if (std::abs(piece) != 1) return false;
    int rank = to / 8;
    return (piece > 0) ? (rank == 0) : (rank == 7);
}

bool Board::isCastlingMove(int from, int to) const {
    int piece = getPiece(from);
    if (std::abs(piece) != 6) return false;
    return std::abs((to % 8) - (from % 8)) == 2;
}

void Board::updateCastlingRightsFromMove(int from, int to, int capturedSquare) {
    if (from == 60 || to == 60) { castleWK = false; castleWQ = false; }
    if (from == 4 || to == 4) { castleBK = false; castleBQ = false; }
    if (from == 63 || to == 63 || capturedSquare == 63) castleWK = false;
    if (from == 56 || to == 56 || capturedSquare == 56) castleWQ = false;
    if (from == 7 || to == 7 || capturedSquare == 7) castleBK = false;
    if (from == 0 || to == 0 || capturedSquare == 0) castleBQ = false;
}

bool Board::leavesKingInCheck(int from, int to, int promoChoice) const {
    Board temp = *this;
    int color = (temp.getPiece(from) > 0) ? WHITE : BLACK;
    int piece = temp.getPiece(from);
    bool isEP = (std::abs(piece) == 1 && to == temp.enPassantTarget);
    bool isCastle = temp.isCastlingMove(from, to);

    int epSq = isEP ? ((color == WHITE) ? to + 8 : to - 8) : -1;
    int rookFrom = -1, rookTo = -1, rookPiece = 0;
    if (isCastle) {
        if (to > from) { rookFrom = (color == WHITE) ? 63 : 7; rookTo = to - 1; }
        else { rookFrom = (color == WHITE) ? 56 : 0; rookTo = to + 1; }
        rookPiece = temp.getPiece(rookFrom);
    }

    temp.setPiece(to, (promoChoice && temp.isPawnPromotion(from, to)) ? ((color == WHITE) ? promoChoice : -promoChoice) : piece);
    temp.setPiece(from, 0);
    if (isEP) temp.setPiece(epSq, 0);
    if (isCastle) { temp.setPiece(rookTo, rookPiece); temp.setPiece(rookFrom, 0); }

    return !temp.isInCheck(color);
}

void Board::applyMove(const Move& m) {
    if (m.wasCastling) {
        setPiece(m.toSquare, m.pieceMoved); setPiece(m.fromSquare, 0);
        setPiece(m.rookTo, m.rookPiece); setPiece(m.rookFrom, 0);
        return;
    }
    setPiece(m.toSquare, m.promotionPiece != 0 ? m.promotionPiece : m.pieceMoved);
    setPiece(m.fromSquare, 0);
    if (m.wasEnPassant) setPiece(m.enPassantCapturedSquare, 0);
}

void Board::revertMove(const Move& m) {
    castleWK = m.castleWKBefore; castleWQ = m.castleWQBefore;
    castleBK = m.castleBKBefore; castleBQ = m.castleBQBefore;
    enPassantTarget = m.enPassantTargetBefore;
    if (m.wasCastling) {
        setPiece(m.fromSquare, m.pieceMoved); setPiece(m.toSquare, 0);
        setPiece(m.rookFrom, m.rookPiece); setPiece(m.rookTo, 0);
        return;
    }
    setPiece(m.fromSquare, m.pieceMoved);
    if (m.wasEnPassant) { setPiece(m.toSquare, 0); setPiece(m.enPassantCapturedSquare, m.pieceCaptured); }
    else { setPiece(m.toSquare, m.pieceCaptured); }
}

std::string Board::squareName(int sq) const {
    return std::string{static_cast<char>('a' + (sq % 8)), static_cast<char>('8' - (sq / 8))};
}

char Board::pieceLetter(int pieceId) const {
    switch (std::abs(pieceId)) {
        case 1: return 'P'; case 2: return 'N'; case 3: return 'B';
        case 4: return 'R'; case 5: return 'Q'; case 6: return 'K';
        default: return '\0';
    }
}

std::string Board::formatMoveNotation(const Move& m) const {
    if (m.wasCastling) return (m.toSquare > m.fromSquare) ? "O-O" : "O-O-O";
    std::string s; char p = pieceLetter(m.pieceMoved);
    if (p != 'K' && p != 'P') s += p;
    bool capture = m.pieceCaptured != 0 || m.wasEnPassant;
    if (capture && p == 'P') s += squareName(m.fromSquare)[0];
    if (capture) s += 'x';
    s += squareName(m.toSquare);
    if (m.promotionPiece != 0) { s += '='; s += pieceLetter(m.promotionPiece); }
    return s;
}

bool Board::isLegalPromotionMove(int from, int to) const {
    if (!isPawnPromotion(from, to)) return false;
    for (int promo : {2, 3, 4, 5}) if (leavesKingInCheck(from, to, promo)) return true;
    return false;
}

bool Board::hasLegalMoves(int color) const {
    int colorIdx = (color == WHITE) ? 0 : 1;
    uint64_t myPieces = colorBB[colorIdx];
    while (myPieces) {
        int from = __builtin_ctzll(myPieces);
        myPieces &= myPieces - 1;
        int piece = getPiece(from);
        int type = std::abs(piece) - 1;
        uint64_t attacks = 0;

        if (type == 0) {
            int dir = (color == WHITE) ? -8 : 8;
            int to1 = from + dir;
            if (to1 >= 0 && to1 < 64 && !(occ & (1ULL << to1))) {
                if (leavesKingInCheck(from, to1, 0)) return true;
            }
            uint64_t i_attacks = pawnAttacks[colorIdx][from] & colorBB[colorIdx ^ 1];
            while (i_attacks) {
                int to = __builtin_ctzll(i_attacks);
                if (leavesKingInCheck(from, to, 0)) return true;
                i_attacks &= i_attacks - 1;
            }
            if (enPassantTarget != -1 && (pawnAttacks[colorIdx][from] & (1ULL << enPassantTarget))) {
                if (leavesKingInCheck(from, enPassantTarget, 0)) return true;
            }
            continue;
        }
        else if (type == 1) attacks = knightAttacks[from];
        else if (type == 2) attacks = getBishopAttacks(from, occ);
        else if (type == 3) attacks = getRookAttacks(from, occ);
        else if (type == 4) attacks = getBishopAttacks(from, occ) | getRookAttacks(from, occ);
        else if (type == 5) attacks = kingAttacks[from];

        attacks &= ~colorBB[colorIdx];
        while (attacks) {
            int to = __builtin_ctzll(attacks);
            attacks &= attacks - 1;
            if (leavesKingInCheck(from, to, 0)) return true;
        }
    }
    return false;
}

bool Board::isCheckmate() const { return isInCheck(turn) && !hasLegalMoves(turn); }
bool Board::isStalemate() const { return !isInCheck(turn) && !hasLegalMoves(turn); }
bool Board::isGameOver() const { return gameResult != 0; }

std::string Board::getStatusText() const {
    if (gameResult == 1) return "Checkmate - White wins";
    if (gameResult == -1) return "Checkmate - Black wins";
    if (gameResult == 2) return "Stalemate - Draw";
    if (isInCheck(turn)) return "Check!";
    return "";
}

bool Board::makeMove(int from, int to, int promoChoice) { return makeMoveWithFullValidation(from, to, promoChoice); }

void Board::undoMove() {
    if (moveHistory.empty()) return;
    Move last = moveHistory.back();
    moveHistory.pop_back();
    if (!moveNotation.empty()) moveNotation.pop_back();
    revertMove(last);
    turn = -turn; gameResult = 0;
}

bool Board::makeMoveWithFullValidation(int from, int to, int promoChoice) {
    if (gameResult != 0) return false;
    int piece = getPiece(from);
    if (piece == 0 || ((piece > 0) != (turn == WHITE))) return false;
    if (isColorMatch(to, turn) && !isCastlingMove(from, to)) return false;

    if (!leavesKingInCheck(from, to, promoChoice)) return false;

    bool isPromo = isPawnPromotion(from, to) && promoChoice != 0;

    Move m{};
    m.fromSquare = from; m.toSquare = to; m.pieceMoved = piece; m.pieceCaptured = getPiece(to);
    m.promotionPiece = isPromo ? ((piece > 0) ? promoChoice : -promoChoice) : 0;
    m.wasCastling = isCastlingMove(from, to); m.wasEnPassant = false; m.enPassantCapturedSquare = -1;
    m.enPassantTargetBefore = enPassantTarget;
    m.castleWKBefore = castleWK; m.castleWQBefore = castleWQ; m.castleBKBefore = castleBK; m.castleBQBefore = castleBQ;

    if (std::abs(piece) == 1 && to == enPassantTarget) {
        m.wasEnPassant = true;
        m.enPassantCapturedSquare = (piece > 0) ? to + 8 : to - 8;
        m.pieceCaptured = getPiece(m.enPassantCapturedSquare);
    }
    if (m.wasCastling) {
        if (to > from) { m.rookFrom = (piece > 0) ? 63 : 7; m.rookTo = to - 1; }
        else { m.rookFrom = (piece > 0) ? 56 : 0; m.rookTo = to + 1; }
        m.rookPiece = getPiece(m.rookFrom);
    }

    applyMove(m);

    updateCastlingRightsFromMove(from, to, m.wasEnPassant ? m.enPassantCapturedSquare : (m.pieceCaptured != 0 ? to : -1));
    enPassantTarget = -1;
    if (std::abs(piece) == 1 && std::abs((to / 8) - (from / 8)) == 2) enPassantTarget = (from + to) / 2;

    moveHistory.push_back(m); moveNotation.push_back(formatMoveNotation(m));
    turn = -turn;
    if (isCheckmate()) gameResult = (turn == WHITE) ? -1 : 1;
    else if (isStalemate()) gameResult = 2;
    return true;
}
