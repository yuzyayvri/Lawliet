#pragma once
#include <string>
#include <vector>
#include <cstddef>

struct EvalParams {
    // Piece values (P, N, B, R, Q)
    int pieceValuesMidgame[5];
    int pieceValuesEndgame[5];

    // PST Tables
    int pawnTableMidgame[64];
    int pawnTableEndgame[64];
    int knightTableMidgame[64];
    int knightTableEndgame[64];
    int bishopTableMidgame[64];
    int bishopTableEndgame[64];
    int rookTableMidgame[64];
    int rookTableEndgame[64];
    int queenTableMidgame[64];
    int queenTableEndgame[64];
    int kingTableMidgame[64];
    int kingTableEndgame[64];

    // Positional / Pawn Terms
    int BishopPairMg, BishopPairEg;
    int DoubledPawnMg, DoubledPawnEg;
    int IsolatedPawnMg, IsolatedPawnEg;
    int BackwardPawnMg, BackwardPawnEg;
    int ConnectedPawnMg, ConnectedPawnEg;
    int ConnectedPawnDefendedMg, ConnectedPawnDefendedEg;
    int ConnectedPawnPhalanxMg, ConnectedPawnPhalanxEg;

    // Passed Pawns
    int PassedPawnRankMg[8];
    int PassedPawnRankEg[8];

    int ConnectedPassedPawnMgBase, ConnectedPassedPawnMgFactor;
    int ConnectedPassedPawnEgBase, ConnectedPassedPawnEgFactor;

    int RookBehindFriendlyPassedPawnMg, RookBehindFriendlyPassedPawnEg;
    int RookBehindEnemyPassedPawnMg, RookBehindEnemyPassedPawnEg;

    int KingProximityToPassedPawnEg;

    int CenterPawnOccupancyMg;
    int PawnAttacksCentralRanksMg;

    int PawnAttackingPieceMg[5];
    int PawnAttackingPieceEg[5];

    // Rook Activity
    int DoubledRooksOnFileMg, DoubledRooksOnFileEg;

    int RookOn7thRankMg, RookOn7thRankEg;
    int RookOn7thRankWithTargetMg, RookOn7thRankWithTargetEg;
    int DoubledRooksOn7thRankMg, DoubledRooksOn7thRankEg;

    int RookOnOpenFileMg, RookOnOpenFileEg;
    int RookOnSemiOpenFileMg, RookOnSemiOpenFileEg;

    // Development
    int UndevelopedMinorPenaltyMg;
    int UndevelopedQueenPenaltyMg;

    // Mobility
    int KnightMobilityMg[9];
    int KnightMobilityEg[9];
    int BishopMobilityMg[14];
    int BishopMobilityEg[14];
    int RookMobilityMg[15];
    int RookMobilityEg[15];
    int QueenMobilityMg[28];
    int QueenMobilityEg[28];

    // Outposts / Defense
    int DefendedMinorBonusMg, DefendedMinorBonusEg;

    int KnightOutpostUnattackableMg, KnightOutpostUnattackableEg;
    int KnightOutpostAttackableMg, KnightOutpostAttackableEg;
    int BishopOutpostDefendedMg, BishopOutpostDefendedEg;

    int UndefendedMinorPenaltyMg, UndefendedMinorPenaltyEg;

    // King safety and positioning
    int KingCentralizationEg;

    int KingFileOpenPenaltyMg, KingFileOpenPenaltyEg;
    int KingFileSemiOpenPenaltyMg, KingFileSemiOpenPenaltyEg;
    int KingFileEnemyMajorAttackPenaltyMg, KingFileEnemyMajorAttackPenaltyEg;

    int KingPawnShieldDistancePenaltiesMg[3];
    int KingPawnShieldDistancePenaltiesEg[3];

    int PawnStormPenaltiesMg[3];

    int KingSubRankPenaltyMg;

    // King zone attack values (Non-linear weights - keep fixed during optimization)
    int KingZoneAttackWeightKnight;
    int KingZoneAttackWeightBishop;
    int KingZoneAttackWeightRook;
    int KingZoneAttackWeightQueen;
    int KingDangerScaleMg;

    int CastleWKMg;
    int CastleWQMg;

    int TempoBonus;
};

extern EvalParams g_Params;

struct ParameterRef {
    std::string name;
    int* ptr;
    int default_value;
};

constexpr size_t EXPECTED_PARAMETER_COUNT = 1010;

std::vector<ParameterRef> get_parameter_references(EvalParams& p);
