#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include "board.hpp"
#include "magic_constants.hpp"
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <cstring>

uint64_t Board::knightAttacks[64];
uint64_t Board::kingAttacks[64];
uint64_t Board::pawnAttacks[2][64];
uint64_t Board::rayMasks[64][8];
uint64_t Board::bishopMagics[64];
uint64_t Board::rookMagics[64];
uint64_t Board::bishopMasks[64];
uint64_t Board::rookMasks[64];
int Board::bishopShift[64];
int Board::rookShift[64];
uint64_t Board::bishopAttacks[64][1 << 9];
uint64_t Board::rookAttacks[64][1 << 12];
bool Board::tablesInitialized = false;

static uint64_t bishopAttacksBruteForce(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int r = sq / 8, f = sq % 8;
    for (int nr = r - 1, nf = f + 1; nr >= 0 && nf < 8; nr--, nf++) {
        uint64_t bit = 1ULL << (nr * 8 + nf); attacks |= bit; if (occ & bit) break;
    }
    for (int nr = r + 1, nf = f - 1; nr < 8 && nf >= 0; nr++, nf--) {
        uint64_t bit = 1ULL << (nr * 8 + nf); attacks |= bit; if (occ & bit) break;
    }
    for (int nr = r - 1, nf = f - 1; nr >= 0 && nf >= 0; nr--, nf--) {
        uint64_t bit = 1ULL << (nr * 8 + nf); attacks |= bit; if (occ & bit) break;
    }
    for (int nr = r + 1, nf = f + 1; nr < 8 && nf < 8; nr++, nf++) {
        uint64_t bit = 1ULL << (nr * 8 + nf); attacks |= bit; if (occ & bit) break;
    }
    return attacks;
}

static uint64_t rookAttacksBruteForce(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int r = sq / 8, f = sq % 8;
    for (int nr = r - 1; nr >= 0; nr--) {
        uint64_t bit = 1ULL << (nr * 8 + f); attacks |= bit; if (occ & bit) break;
    }
    for (int nr = r + 1; nr < 8; nr++) {
        uint64_t bit = 1ULL << (nr * 8 + f); attacks |= bit; if (occ & bit) break;
    }
    for (int nf = f - 1; nf >= 0; nf--) {
        uint64_t bit = 1ULL << (r * 8 + nf); attacks |= bit; if (occ & bit) break;
    }
    for (int nf = f + 1; nf < 8; nf++) {
        uint64_t bit = 1ULL << (r * 8 + nf); attacks |= bit; if (occ & bit) break;
    }
    return attacks;
}

int Board::pawnTableMidgame[64] = {0};
int Board::knightTableMidgame[64] = {0};
int Board::bishopTableMidgame[64] = {0};
int Board::rookTableMidgame[64] = {0};
int Board::queenTableMidgame[64] = {0};
int Board::kingTableMidgame[64] = {0};
int Board::pawnTableEndgame[64] = {0};
int Board::knightTableEndgame[64] = {0};
int Board::bishopTableEndgame[64] = {0};
int Board::rookTableEndgame[64] = {0};
int Board::queenTableEndgame[64] = {0};
int Board::kingTableEndgame[64] = {0};

int Board::pieceValuesMidgame[6] = {0};
int Board::pieceValuesEndgame[6] = {0};

const int* Board::pstMidgame[6] = {nullptr};
const int* Board::pstEndgame[6] = {nullptr};

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
    halfMoveClock = 0;

    loadParams();
    initAttackTables();
    reset();
}

void Board::reset() {
    loadFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::loadParams() {
    for (int i = 0; i < 5; ++i) {
        pieceValuesMidgame[i] = g_Params.pieceValuesMidgame[i];
        pieceValuesEndgame[i] = g_Params.pieceValuesEndgame[i];
    }
    pieceValuesMidgame[5] = 20000;
    pieceValuesEndgame[5] = 20000;

    std::memcpy(pawnTableMidgame, g_Params.pawnTableMidgame, sizeof(pawnTableMidgame));
    std::memcpy(pawnTableEndgame, g_Params.pawnTableEndgame, sizeof(pawnTableEndgame));
    std::memcpy(knightTableMidgame, g_Params.knightTableMidgame, sizeof(knightTableMidgame));
    std::memcpy(knightTableEndgame, g_Params.knightTableEndgame, sizeof(knightTableEndgame));
    std::memcpy(bishopTableMidgame, g_Params.bishopTableMidgame, sizeof(bishopTableMidgame));
    std::memcpy(bishopTableEndgame, g_Params.bishopTableEndgame, sizeof(bishopTableEndgame));
    std::memcpy(rookTableMidgame, g_Params.rookTableMidgame, sizeof(rookTableMidgame));
    std::memcpy(rookTableEndgame, g_Params.rookTableEndgame, sizeof(rookTableEndgame));
    std::memcpy(queenTableMidgame, g_Params.queenTableMidgame, sizeof(queenTableMidgame));
    std::memcpy(queenTableEndgame, g_Params.queenTableEndgame, sizeof(queenTableEndgame));
    std::memcpy(kingTableMidgame, g_Params.kingTableMidgame, sizeof(kingTableMidgame));
    std::memcpy(kingTableEndgame, g_Params.kingTableEndgame, sizeof(kingTableEndgame));

    pstMidgame[0] = pawnTableMidgame;
    pstMidgame[1] = knightTableMidgame;
    pstMidgame[2] = bishopTableMidgame;
    pstMidgame[3] = rookTableMidgame;
    pstMidgame[4] = queenTableMidgame;
    pstMidgame[5] = kingTableMidgame;

    pstEndgame[0] = pawnTableEndgame;
    pstEndgame[1] = knightTableEndgame;
    pstEndgame[2] = bishopTableEndgame;
    pstEndgame[3] = rookTableEndgame;
    pstEndgame[4] = queenTableEndgame;
    pstEndgame[5] = kingTableEndgame;
}

bool Board::loadFen(const std::string& fen) {
    // Completely clear board arrays and reset evaluation accumulators to 0
    std::memset(pieceBB, 0, sizeof(pieceBB));
    std::memset(colorBB, 0, sizeof(colorBB));
    occ = 0;
    mgPst = 0;
    egPst = 0;
    gameResult = 0;
    halfMoveClock = 0;
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

    // Parse half-move clock (50-move rule counter)
    std::string hmPart;
    halfMoveClock = 0;
    if (iss >> hmPart) {
        halfMoveClock = std::stoi(hmPart);
    }

    loadParams();
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

    // Initialize magic bitboards using precomputed constants
    for (int sq = 0; sq < 64; sq++) {
        bishopMasks[sq] = BISHOP_MASKS[sq];
        bishopShift[sq] = BISHOP_SHIFT[sq];
        bishopMagics[sq] = BISHOP_MAGICS[sq];
        int bSize = 1 << (64 - bishopShift[sq]);
        for (int i = 0; i < bSize; i++) {
            uint64_t occ = 0;
            uint64_t m = bishopMasks[sq];
            int idx = i;
            while (m) {
                uint64_t bit = m & -m;
                m ^= bit;
                if (idx & 1) occ |= bit;
                idx >>= 1;
            }
            bishopAttacks[sq][(occ * bishopMagics[sq]) >> bishopShift[sq]] = bishopAttacksBruteForce(sq, occ);
        }

        rookMasks[sq] = ROOK_MASKS[sq];
        rookShift[sq] = ROOK_SHIFT[sq];
        rookMagics[sq] = ROOK_MAGICS[sq];
        int rSize = 1 << (64 - rookShift[sq]);
        for (int i = 0; i < rSize; i++) {
            uint64_t occ = 0;
            uint64_t m = rookMasks[sq];
            int idx = i;
            while (m) {
                uint64_t bit = m & -m;
                m ^= bit;
                if (idx & 1) occ |= bit;
                idx >>= 1;
            }
            rookAttacks[sq][(occ * rookMagics[sq]) >> rookShift[sq]] = rookAttacksBruteForce(sq, occ);
        }
    }
}

uint64_t Board::getRookAttacks(int sq, uint64_t occ) {
    uint64_t masked = occ & rookMasks[sq];
    return rookAttacks[sq][(masked * rookMagics[sq]) >> rookShift[sq]];
}

uint64_t Board::getBishopAttacks(int sq, uint64_t occ) {
    uint64_t masked = occ & bishopMasks[sq];
    return bishopAttacks[sq][(masked * bishopMagics[sq]) >> bishopShift[sq]];
}

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
            int promoRank = (color == WHITE) ? 0 : 7;
            int to1 = from + dir;
            if (to1 >= 0 && to1 < 64 && !(occ & (1ULL << to1))) {
                if (to1 / 8 == promoRank) {
                    for (int promo : {2, 3, 4, 5}) { if (leavesKingInCheck(from, to1, promo)) return true; }
                } else {
                    if (leavesKingInCheck(from, to1, 0)) return true;
                }
            }
            uint64_t i_attacks = pawnAttacks[colorIdx][from] & colorBB[colorIdx ^ 1];
            while (i_attacks) {
                int to = __builtin_ctzll(i_attacks);
                if (to / 8 == promoRank) {
                    for (int promo : {2, 3, 4, 5}) { if (leavesKingInCheck(from, to, promo)) return true; }
                } else {
                    if (leavesKingInCheck(from, to, 0)) return true;
                }
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
        else if (type == 5) {
            attacks = kingAttacks[from];
            if (color == WHITE) {
                if (from == 60) {
                    if (castleWK && !(occ & ((1ULL<<61)|(1ULL<<62))) && !isInCheck(WHITE) && !isSquareAttacked(61, BLACK) && !isSquareAttacked(62, BLACK)) {
                        if (leavesKingInCheck(60, 62, 0)) return true;
                    }
                    if (castleWQ && !(occ & ((1ULL<<57)|(1ULL<<58)|(1ULL<<59))) && !isInCheck(WHITE) && !isSquareAttacked(59, BLACK) && !isSquareAttacked(58, BLACK)) {
                        if (leavesKingInCheck(60, 58, 0)) return true;
                    }
                }
            } else {
                if (from == 4) {
                    if (castleBK && !(occ & ((1ULL<<5)|(1ULL<<6))) && !isInCheck(BLACK) && !isSquareAttacked(5, WHITE) && !isSquareAttacked(6, WHITE)) {
                        if (leavesKingInCheck(4, 6, 0)) return true;
                    }
                    if (castleBQ && !(occ & ((1ULL<<1)|(1ULL<<2)|(1ULL<<3))) && !isInCheck(BLACK) && !isSquareAttacked(3, WHITE) && !isSquareAttacked(2, WHITE)) {
                        if (leavesKingInCheck(4, 2, 0)) return true;
                    }
                }
            }
        }

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

bool Board::checkInvariants() const {
    if (__builtin_popcountll(pieceBB[5]) != 1) return false;
    if (__builtin_popcountll(pieceBB[11]) != 1) return false;

    uint64_t wPieces = 0, bPieces = 0, expectedOcc = 0;
    for (int i = 0; i < 6; ++i) { wPieces |= pieceBB[i]; expectedOcc |= pieceBB[i]; }
    for (int i = 6; i < 12; ++i) { bPieces |= pieceBB[i]; expectedOcc |= pieceBB[i]; }

    if (occ != expectedOcc) return false;
    if (colorBB[0] != wPieces) return false;
    if (colorBB[1] != bPieces) return false;
    if (colorBB[0] & colorBB[1]) return false;

    if (turn != WHITE && turn != BLACK) return false;
    if (enPassantTarget != -1 && (enPassantTarget < 0 || enPassantTarget > 63)) return false;
    return true;
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

    // Update 50-move rule clock
    if (std::abs(piece) == 1 || m.pieceCaptured != 0) halfMoveClock = 0;
    else halfMoveClock++;

    moveHistory.push_back(m); moveNotation.push_back(formatMoveNotation(m));
    turn = -turn;
    if (isCheckmate()) gameResult = (turn == WHITE) ? -1 : 1;
    else if (isStalemate()) gameResult = 2;
    return true;
}
