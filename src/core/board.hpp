#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "parameters.hpp"

struct Move {
    int fromSquare = -1;
    int toSquare = -1;
    int pieceMoved = 0;
    int pieceCaptured = 0;
    int promotionPiece = 0;
    bool wasCastling = false;
    int rookFrom = -1;
    int rookTo = -1;
    int rookPiece = 0;
    bool wasEnPassant = false;
    int enPassantCapturedSquare = -1;
    int enPassantTargetBefore = -1;
    bool castleWKBefore = false;
    bool castleWQBefore = false;
    bool castleBKBefore = false;
    bool castleBQBefore = false;

    bool operator==(const Move& other) const {
        return fromSquare == other.fromSquare &&
        toSquare == other.toSquare &&
        promotionPiece == other.promotionPiece;
    }
};

class Board {
    friend class Lawliet;
public:
    uint64_t pieceBB[12]; // P, N, B, R, Q, K for White (0-5) and Black (6-11)
    uint64_t colorBB[2];  // White (0), Black (1)
    uint64_t occ;         // Combined occupancy

    std::vector<Move> moveHistory;
    std::vector<std::string> moveNotation;
    int turn;
    int enPassantTarget;
    bool castleWK, castleWQ, castleBK, castleBQ;
    int gameResult;

    int mgPst;
    int egPst;

    int halfMoveClock;

    Board();
    void reset();
    bool loadFen(const std::string& fen);
    bool makeMove(int from, int to, int promoChoice = 0);
    void undoMove();

    inline int getPiece(int sq) const {
        uint64_t bit = 1ULL << sq;
        if (!(occ & bit)) return 0;
        for (int i = 0; i < 6; ++i) {
            if (pieceBB[i] & bit) return i + 1;       // White pieces: 1 to 6
            if (pieceBB[i + 6] & bit) return -(i + 1); // Black pieces: -1 to -6
        }
        return 0;
    }

    bool isColorMatch(int square, int color) const;
    bool isSquareAttacked(int square, int attackerColor) const;
    inline int getPieceType(int pieceId) const { return std::abs(pieceId) - 1; }
    bool isPawnPromotion(int from, int to) const;
    bool isCastlingMove(int from, int to) const;
    bool makeMoveWithFullValidation(int from, int to, int promoChoice = 0);
    bool isInCheck(int color) const;
    bool isCheckmate() const;
    bool isStalemate() const;
    bool isGameOver() const;
    bool hasLegalMoves(int color) const;
    bool isLegalPromotionMove(int from, int to) const;
    std::string getStatusText() const;
    std::string formatMoveNotation(const Move& m) const;
    int findKing(int color) const;
    bool leavesKingInCheck(int from, int to, int promoChoice) const;

    static constexpr int WHITE = 1;
    static constexpr int BLACK = -1;

    static uint64_t knightAttacks[64];
    static uint64_t kingAttacks[64];
    static uint64_t pawnAttacks[2][64];
    static uint64_t rayMasks[64][8]; // Precomputed rays for sliding pieces
    static bool tablesInitialized;
    static void initAttackTables();
    static uint64_t getRookAttacks(int sq, uint64_t occ);
    static uint64_t getBishopAttacks(int sq, uint64_t occ);
    void loadParams();

    static int pawnTableMidgame[64];
    static int knightTableMidgame[64];
    static int bishopTableMidgame[64];
    static int rookTableMidgame[64];
    static int queenTableMidgame[64];
    static int kingTableMidgame[64];
    static int pawnTableEndgame[64];
    static int knightTableEndgame[64];
    static int bishopTableEndgame[64];
    static int rookTableEndgame[64];
    static int queenTableEndgame[64];
    static int kingTableEndgame[64];

    static int pieceValuesMidgame[6];
    static int pieceValuesEndgame[6];

    static const int* pstMidgame[6];
    static const int* pstEndgame[6];

private:
    void applyMove(const Move& m);
    void revertMove(const Move& m);
    void updateCastlingRightsFromMove(int from, int to, int capturedSquare);
    std::string squareName(int sq) const;
    char pieceLetter(int pieceId) const;

    inline void setPiece(int sq, int piece) {
        uint64_t bit = 1ULL << sq;
        int oldPiece = getPiece(sq);

        if (oldPiece != 0) {
            int oldType = std::abs(oldPiece) - 1;
            bool oldIsWhite = (oldPiece > 0);
            int oldIdx = oldIsWhite ? sq : ((7 - (sq / 8)) * 8 + (sq % 8));
            int oldMg = pieceValuesMidgame[oldType] + pstMidgame[oldType][oldIdx];
            int oldEg = pieceValuesEndgame[oldType] + pstEndgame[oldType][oldIdx];
            if (oldIsWhite) {
                mgPst -= oldMg;
                egPst -= oldEg;
            } else {
                mgPst += oldMg;
                egPst += oldEg;
            }
        }

        for (int i = 0; i < 12; ++i) pieceBB[i] &= ~bit;
        colorBB[0] &= ~bit; colorBB[1] &= ~bit;

        if (piece == 0) {
            occ &= ~bit;
            return;
        }

        occ |= bit;
        if (piece > 0) {
            pieceBB[piece - 1] |= bit;
            colorBB[0] |= bit;
        } else {
            pieceBB[(-piece) + 5] |= bit;
            colorBB[1] |= bit;
        }

        int newType = std::abs(piece) - 1;
        bool newIsWhite = (piece > 0);
        int newIdx = newIsWhite ? sq : ((7 - (sq / 8)) * 8 + (sq % 8));
        int newMg = pieceValuesMidgame[newType] + pstMidgame[newType][newIdx];
        int newEg = pieceValuesEndgame[newType] + pstEndgame[newType][newIdx];
        if (newIsWhite) {
            mgPst += newMg;
            egPst += newEg;
        } else {
            mgPst -= newMg;
            egPst -= newEg;
        }
    }
};
