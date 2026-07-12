# Lawliet Chess Engine — Project Knowledge Base

> This file is for AI coding agents. It documents project architecture, implementation details, known issues, and hard-learned lessons.

> Update this file appropriately after major changes to maintain relevance.

---

## 1. PROJECT OVERVIEW

**Lawliet** is a UCI-compatible chess engine written in **C++20** by Yuzy. It uses:
- **Stockfish 13 HalfKP NNUE** for evaluation (primary path when loaded)
- **Hand-Crafted Evaluation (HCE)** as fallback (tapered MG/EG with ~1010 parameters)
- **Alpha-beta search** with PVS, aspiration windows, LMR, singular extensions, and multiple pruning techniques
- **Magic bitboards** for sliding piece attack generation
- **Polyglot opening book** support
- **Lazy SMP** multithreading
- **Static Exchange Evaluation (SEE)** for move ordering and pruning
- **Texel tuning** for HCE parameters

**Versions**: 1st gen = Intuition, 2nd gen (planned) = Deduction, 3rd gen (planned) = Mastermind.

---

## 2. BUILD SYSTEM

### CMake Configuration (`CMakeLists.txt`)
- **Minimum CMake**: 3.15
- **C++ Standard**: 20 (required)
- **Dependencies**: SFML 3 (Graphics, Window, System) — only for the legacy GUI sandbox
- **Compiler targets**: `x86-64` with AVX2, BMI, BMI2, LZCNT, POPCNT

### Build Commands
```bash
mkdir build && cd build
cmake ..
make chess_app           # Main engine binary
make texel_tuner         # HCE parameter tuning tool
make stockfish_annotator # Standalone EPD annotator
```

### Compiler Flags (hardcoded in source)
```cpp
#pragma GCC optimize("O3,unroll-loops,fast-math")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
```
These pragmas are at the top of `lawliet.cpp`, `nnue.cpp`, `board.cpp`, and `interface.cpp`.

### Build Targets
| Target | Description | Sources |
|--------|-------------|---------|
| `chess_app` | Main engine + optional GUI | All `src/*.cpp` except `tuner_main.cpp` and `annotator_main.cpp` |
| `texel_tuner` | HCE parameter tuner | All `src/*.cpp` except `main.cpp` and `annotator_main.cpp` |
| `stockfish_annotator` | EPD annotator | Only `annotator_main.cpp` (standalone, no SFML) |

---

## 3. FILE STRUCTURE

```
src/
├── main.cpp                    # Entry point: UCI mode or GUI sandbox
├── core/
│   ├── board.hpp / board.cpp   # Board representation, magic bitboards, FEN, move gen
│   ├── parameters.hpp / .cpp   # HCE evaluation parameters (1010 params, Texel-tuned)
│   ├── engine_options.hpp      # Threads, Hash, BookPath, NNUEPath, etc.
│   └── magic_constants.hpp     # Precomputed magic bitboard constants
├── ai/
│   ├── lawliet.hpp / lawliet.cpp # Engine core: search, evaluation, NNUE orchestration
│   ├── nnue.hpp / nnue.cpp     # Stockfish 13 HalfKP NNUE (AVX2 forward pass)
│   ├── uci.hpp / uci.cpp       # UCI protocol handler
│   └── time_manager.hpp / .cpp # Time management (soft/hard limits, movetime, depth)
├── create_nnue_structure.sh    # Helper script for setting up NNUE weight files
├── ui/
│   └── interface.hpp / .cpp    # Legacy SFML GUI sandbox (not recommended)
└── tools/
    ├── texel_tuner/
    │   └── tuner_main.cpp      # Texel tuning tool for HCE parameters
    └── stockfish_annotator/
        └── annotator_main.cpp  # EPD annotation tool (subprocess-based)
```

---

## 4. BOARD REPRESENTATION

### Square Indexing
- **Lawliet convention**: `a8=0, b8=1, ..., h1=63` (row-major, rank 0 = rank 8)
- **Stockfish convention**: `A1=0, B1=1, ..., H8=63`
- Conversion: `toSfSq(sq) = (7 - sq/8) * 8 + sq%8`

### Piece Encoding
```
White: P=1, N=2, B=3, R=4, Q=5, K=6
Black: P=-1, N=-2, B=-3, R=-4, Q=-5, K=-6
```

### Bitboard Arrays
```cpp
uint64_t pieceBB[12]; // [P,N,B,R,Q,K] white (0-5) and black (6-11)
uint64_t colorBB[2];  // White=0, Black=1
uint64_t occ;         // Combined occupancy
```

### KEY: `makeMove()` vs `makeMoveWithFullValidation()`
- `makeMove()` delegates to `makeMoveWithFullValidation()` — **always validates legality**
- For search, the engine uses `doMove()`/`undoMove()` directly (no validation) with pseudo-legal move generation and legality check after move via `board.isInCheck(-board.turn)`

### Move Structure
```cpp
struct Move {
    int fromSquare, toSquare;
    int pieceMoved, pieceCaptured, promotionPiece;
    bool wasCastling, wasEnPassant;
    int rookFrom, rookTo, rookPiece;
    int enPassantCapturedSquare, enPassantTargetBefore;
    bool castleWKBefore, castleWQBefore, castleBKBefore, castleBQBefore;
};
```

---

## 5. EVALUATION

### NNUE (Primary Path)
- **Architecture**: Stockfish 13 HalfKP(Friend) `41024 -> 256x2 -> 32 -> 32 -> 1`
- **Weight file**: `nn-62ef826d1a6d.nnue` (Stockfish 13 net)
- **Perspective ordering**: `[side_to_move, ~side_to_move]` (NOT the older `[white, black]`)
- **CReLU**: Clamp FT outputs to `[0, 127]`, hidden layers to `x >> 6` then `[0, 127]`
- **Output scaling**: `raw_score / FV_SCALE` where `FV_SCALE = 16`
- **Value scale**: NNUE produces values ~3x larger than HCE. E.g., a position NNUE evaluates as +57 cp might be ~15 cp under HCE. This is why search parameters were retuned.

#### Auto-Search for NNUE Weight File
The `loadNNUE()` function in `lawliet.cpp` tries these fallback paths when the primary path fails:
1. The configured path (as given)
2. Just the filename (current working directory)
3. `build/<filename>`
4. `../<filename>`
5. `../build/<filename>`
6. `src/ai/<filename>`
7. `../src/ai/<filename>`
8. `../../<filename>`
9. `<exe_dir>/<filename>` (via `/proc/self/exe` + `readlink()`)
10. `<exe_dir_parent>/<filename>`

#### Incremental NNUE Updates
- Search maintains an **accumulator stack** (`nnueAccStack[128]`) updated via `doMove()`
- King moves and castling trigger **full FT rebuild** (king square changes for all features)
- Non-king moves use **SIMD diff updates**: subtract old feature column, add new feature column
- Captures: subtract captured piece's feature column
- Promotions: subtract pawn features, add promotion piece features
- `evaluateBoard()` with `ctx->nnueAccStackIdx >= 0` takes the fast incremental path
- Cold starts (index == -1) or stack overflow fall back to full rebuild

#### ⚠️ CRITICAL BUG HISTORY: NNUE Incremental Updates
Earlier versions had severe bugs in the accumulator stack management:
- **Duplicate `board.applyMove()`** in `doMove()` corrupting board state
- **Stack index off-by-one** causing incorrect accumulator reuse
- **King move handling** using wrong piece indices for feature extraction
- These were fixed and the match result went from **0.278 → 0.795** vs HCE baseline

### HCE (Fallback Path)
- Tapered evaluation: `(mg * phase + eg * (24 - phase)) / 24 + TempoBonus`
- **TempoBonus = 15** — added to every static evaluation from the side-to-move perspective. This ~+15 cp bonus for the side to move affects search stability and pruning decisions.
- Phase: 0 (endgame) to 24 (opening), based on non-pawn material
- **~1010 parameters** in `g_Params` (`parameters.hpp` / `parameters.cpp`), Texel-tuned
- Key evaluation components:
  - Piece-square tables (PSTs) for MG and EG
  - Pawn structure (doubled, isolated, backward, connected, passed)
  - Mobility (safe squares for knights, bishops, rooks, queens)
  - King safety (pawn shield, storm penalties, king zone attacks)
  - King zone danger (non-linear weighting via `danger^2 * scale`)
  - Opposite-colored bishop drawish scaling (50% EG reduction)
  - Bishop pair, rook on 7th, open files, outposts, development penalties
  - Static Evaluation Correction History (CorrHist) — 16-bit indexed per pawn hash

---

## 6. SEARCH ALGORITHM

### Architecture
- **Iterative Deepening** with **Principal Variation Search** (PVS)
- **Aspiration Windows** at root
- **Lazy SMP** for multithreading

### Search Constants
```cpp
INF = 10000000;        // Infinity (mate scores are ±(INF - ply))
MAX_PLY = 128;         // Main search depth
MAX_QDEPTH = 64;       // Quiescence search depth
MAX_TOTAL_PLY = 202;   // Combined limit
```

### Key Parameters (NNUE-Tuned Values)
The search parameters were re-tuned when NNUE was added because NNUE produces values ~3x larger than HCE:

| Parameter | HCE (Original) | NNUE (Current) | Scaling |
|-----------|:--------------:|:--------------:|:-------:|
| **Aspiration window** | `16 + 5*depth` | `40 + 10*depth` | ~2x |
| **ProbCut margin** | `beta + 1000` | `beta + 2000` | 2x |
| **Delta pruning** | `+200` | `+350` | 1.75x |
| **Razoring margin** | `2500 * depth` | `4500 * depth` | 1.8x |
| **RFP margin** | `800 * depth` | `1500 * depth` | 1.9x |
| **Null move divisor** | `/200` | `/300` | 1.5x |

**⚠️ IMPORTANT**: These parameters are optimized for NNUE's value scale. The engine MUST have NNUE loaded to play well. Without NNUE (e.g., missing weight file), the search parameters are too aggressive for HCE values, and the engine will play **much weaker** (~0.278 score vs proper HCE).

### Search Techniques

#### Aspiration Windows
```cpp
int window = 40 + depth * 10; // NNUE-scaled
```
Fail-highs trigger re-search with doubled window, then full window if needed.

#### Null Move Pruning (NMP)
- Depth >= 3, non-PV, not in check, has non-pawn material, `staticEval >= beta`
- Reduction: `R = 3 + depth/6 + min(3, (staticEval - beta) / 300)`
- Uses adaptive verification reduction

#### Reverse Futility Pruning (RFP)
- Depth <= 6, non-PV, not in check
- Margin: `1500 * depth`
- If `staticEval - margin >= beta`, return `staticEval - margin`

#### Razoring
- Depth <= 1, not in check, `staticEval + 2500 * depth < alpha`
- Run QS with reduced window to verify hopelessness
- Margin: `4500 * depth`

#### ProbCut
- Depth >= 4, non-PV, `abs(beta) < INF`
- Run reduced-depth search with `beta + 2000`
- Cuts off if confirmed

#### Singular Extensions
- Depth >= 5 (ply<=1) or depth >= 8 (deeper)
- TT move exists with TT depth within margin
- Search without the TT move; if score drops below `singularBeta`, extend by 1
- Double extension possible for PV nodes at depth >= 12

#### Late Move Reductions (LMR)
- LMR table: `log(d) * log(m) / 1.3` with NNUE-specific tuning
- History-based reduction adjustment
- Capture reductions allowed (non-check only)
- Re-searches at full depth when LMR score exceeds alpha

#### Move Ordering
1. **TT move** (highest priority)
2. **Winning captures** (SEE >= 0, MVV-LVA ordered)
3. **Promotions**
4. **Killer moves** (2 per ply)
5. **Countermove heuristic** (based on previous ply's move)
6. **History heuristic** (piece-to-square table)
7. **Continuation history** (based on previous piece moved)
8. **Losing captures** (SEE < 0, lowest priority)

### Quiescence Search
- Stand-pat (static eval) if not in check
- Delta pruning: `staticEval + pieceValue + 350 <= alpha` skip capture
- SEE-based bad capture pruning (loser exchanges skipped)
- Checks generate all legal moves; non-checks generate only captures + promotions

### Transposition Table
- Size: `1 << 23` default (scales with Hash option, ~8 bytes per entry)
- `TTEntry`: 16 bytes (two `atomic<uint64_t>` for key and data)
- Packed data: score(32), depth(7), flag(2), fromSq(6), toSq(6), promo(4), age(7)
- Replacement: generational with depth-preference (keep deeper entries of same age, replace older entries)
- Aging: `ttAge` byte incremented per search; younger entries preferred for replacement

### Time Management (`TimeManager`)
- **Soft limit**: `time/divisor + inc*0.8` (divisor = min(movestogo+2, 40) or 25)
- **Hard limit**: `max(alloc*2, time*0.10)` — absolute ceiling
- Root-level check: currentDepth > 6 && elapsed >= 50% of alloc triggers stop (in non-movetime mode)
- Depth limit: `maxDepthLimit` for `depth` and `movetime` searches
- `infinite` flag freezes stop checks entirely

---

## 7. UCI PROTOCOL

### UCI Options
```
Threads:       spin default 1 min 1 max 1024
Hash:          spin default 128 min 1 max 1048576
Clear Hash:    button
Ponder:        check default false
BookPath:      string default book.bin
Move Overhead: spin default 10 min 0 max 10000
NNUEPath:      string default nn-62ef826d1a6d.nnue
```

### Auto-Detection of UCI Mode
`main.cpp` checks:
1. `--uci` flag on command line → UCI mode
2. `isatty(STDIN_FILENO)` → if stdin is not a TTY, automatically enter UCI mode (for GUIs that pipe input)

### `eval` Command (Non-Standard Extension)
```bash
# Shows the current static evaluation:
eval
```
Output: `info string NNUE evaluation: 57 cp` (also shows HCE eval if NNUE loaded)

### Opening Book
- Polyglot binary format (`book.bin`)
- Default UCI `BookPath` is `book.bin` (relative path), but the actual book lives at `src/ai/book.bin`
- The `OpeningBook` class auto-searches fallback paths: `book.bin`, `../book.bin`, `../../book.bin`, `bin/book.bin`, etc.
- Best practice: either `option.BookPath=/full/path/to/book.bin` or copy `book.bin` to the build directory
- Loaded at engine startup via `BookPath` option

### NNUE Loading
- Attempted on `isready` via `options.nnuePath`
- Can be reloaded via `setoption name NNUEPath value <path>`
- If NNUE fails to load, engine silently falls back to HCE (but with NNUE-scaled search parameters!)

---

## 8. KNOWN ISSUES & GOTCHAS

### ⚠️ CRITICAL: NNUEPath Must Be Explicit in cutechess-cli
When running matches with cutechess-cli, you MUST specify `option.NNUEPath=<path>` and `option.BookPath=<path>`:
```bash
cutechess-cli -engine cmd=chess_app proto=uci \
    option.NNUEPath=/path/to/nn-62ef826d1a6d.nnue \
    option.BookPath=/path/to/book.bin
```
Without NNUEPath, the engine falls back to HCE with NNUE-scaled search parameters, playing extremely poorly (~0.278 score vs proper HCE).

### ⚠️ Search Parameters Are NNUE-Specific
The search margins (aspiration, RFP, razoring, probcut, delta, null move) are tuned for NNUE's ~3x larger value scale. If you need to run with HCE only, these parameters should be reverted to HCE values.

### ⚠️ NNUE Auto-Search May Not Cover All Environments
The `readlink("/proc/self/exe", ...)` fallback depends on `/proc` being available (**Linux only**). Won't work on macOS or Windows. The fallback searches common paths but may miss unusual directory layouts. The `#include <unistd.h>` is POSIX, but `/proc` is Linux-specific.

### ⚠️ Opening Book Default Path May Not Find the Book
The default UCI BookPath is `book.bin` (relative). If running from outside the build directory (e.g., cutechess-cli may run from $HOME), the book won't be found. Always specify `option.BookPath=<absolute_path>` in match commands.

### ⚠️ NNUE Accumulator Stack
The `nnueAccStackIdx` is set to `-1` (invalid) initially. The first `evaluateBoard` call after `doMove` will trigger a full FT rebuild. The stack has 128 slots; exceeding this falls back to non-cached evaluation.

### ⚠️ Debug Checks in DoMove
```cpp
#ifndef NDEBUG
    if (!board.checkInvariants()) { std::abort(); }
#endif
```
These invariants check that each side has exactly one king, occupancy matches piece bitboards, etc. Useful for debugging but may abort on corruption.

### ⚠️ SFML UI is a Legacy Artifact
The GUI sandbox (`interface.cpp`) is from Lawliet's early days and is not recommended. The UCI mode is the primary interface.

---

## 9. MATCH HISTORY & TESTING WORKFLOW

### Historical Match Results (NNUE vs HCE)

| Match | NNUE Score | Elo | Notes |
|-------|:----------:|:---:|:------|
| Original (bugs + missing NNUEPath) | **0.278** | -174 | NNUE wasn't actually loaded! |
| Fixed bugs + tuned params + NNUEPath set | **0.795** | +235±70 | SPRT LLR=2.62 (near significance) |

The huge jump was caused by two factors:
1. **Bug fixes**: Duplicate `board.applyMove()`, accumulator stack index issues, king move handling
2. **Missing NNUEPath** in cutechess-cli: the earlier 0.278 match was testing HCE-with-NNUE-parameters vs proper HCE

### Key Lesson: NNUEPath Must Be Explicit
Always pass `option.NNUEPath=<absolute_path>` and `option.BookPath=<absolute_path>` to cutechess-cli. Without this, the engine silently falls back to HCE with NNUE-scaled search parameters and plays terribly.

### Correct Match Command Template
```bash
cutechess-cli -tournament gauntlet -rounds 100 -games 2 -repeat -recover \
  -concurrency 4 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -pgnout /path/to/results.pgn min \
  -engine name="Lawliet-NNUE" cmd=/path/to/build/chess_app proto=uci \
    option.NNUEPath=/path/to/nn-62ef826d1a6d.nnue \
    option.BookPath=/path/to/book.bin \
  -engine name="Lawliet-HCE" cmd=/path/to/baseline/build/chess_app proto=uci \
    option.BookPath=/path/to/book.bin \
  -each tc=10+0.1 option.Threads=1 option.Hash=256
```

#### Flag Breakdown
| Flag | What it does |
|------|-------------|
| `-tournament gauntlet` | One engine vs another (not round-robin) |
| `-rounds N` | Number of rounds (each round = `-games 2` games) |
| `-games 2` | 2 games per round (each side once) |
| `-repeat` | Repeat same opening positions across rounds |
| `-recover` | Restart crashed engines automatically |
| `-concurrency N` | Number of parallel games (set to number of cores) |
| `-sprt elo0=0 elo1=10 ...` | Sequential Probability Ratio Test — stops early if result is statistically significant |
| `-pgnout file.pgn min` | Save games, `min` = minimal info only |
| `-each tc=10+0.1 ...` | Apply to both engines: time control, options |
| `option.Threads=1` | Single-threaded for reproducible testing |
| `option.Hash=256` | 256 MB hash table |

#### SPRT Interpretation
- **LLR >= 2.94**: H1 accepted (new version is stronger with 95% confidence)
- **LLR <= -2.94**: H0 accepted (not stronger)
- **Between**: Continue testing (more games needed)
- The test above gave LLR=2.62 after 100 games — very close to significance

### Baseline Build
- **Old HCE-only version**: `/home/yuzy/Lawliet_baseline/Lawliet/build/chess_app`
- **Current baseline** (after each upgrade): The `/home/yuzy/Lawliet_baseline/Lawliet/` directory is updated by copying the current project after a validated improvement. The current baseline is commit `f2c966e`.

### Match History

| Match | Score | Elo | Notes |
|-------|:-----:|:---:|:------|
| Search efficiency vs baseline (50r) | **0.567** | +47 | 97 games, 23-10-64, optimistic outlier |
| Search efficiency vs baseline (SPRT) | **0.490** | -7 | 308 games, no regression confirmed |
| QS staticEval fix vs baseline | **0.480** | -14 | 153 games, no regression confirmed |
| NMP /600 vs baseline | **0.487** | -9 | 866 games — clear regression from widening NMP divisor |
| Revert all changes vs baseline | **0.543** | +30 | 94 games — regression eliminated, code matches baseline exactly |

**Key takeaway**: The NMP divisor change from `/300` to `/600` caused a real regression (~9 Elo). The QS staticEval fix and negamax optimization were individually SPRT-validated as no-regression, but their combined effect was inconclusive. All changes were ultimately reverted to restore baseline parity.

#### White/Black Asymmetry Analysis
The 1102-game match showed a ~0.123 swing between White and Black performance for both engines:
- INDEV White: 0.564 vs INDEV Black: 0.441 (Δ = 0.123)
- OLD White: 0.559 vs OLD Black: 0.436 (Δ = 0.123)

This asymmetry is **identical** between both engines, confirming it is the **normal first-move advantage**, not a code bug.

**Key lesson**: When investigating suspected color asymmetries, compare the baseline's color split first. If both engines show the same swing, the asymmetry is inherent to chess, not your changes.

#### TT Static Eval Caching (Search Efficiency Upgrade — July 2026)
The TT entry was repacked to store a **16-bit static evaluation** alongside the search score:

| Bits | Field |
|:----:|-------|
| 0-17 | score (18-bit signed, ±131071) |
| 18-23 | depth (6 bits, 0-63) |
| 24-25 | flag (2 bits) |
| 26-31 | fromSq (6 bits, 0-63) |
| 32-37 | toSq (6 bits, 0-63) |
| 38-41 | promotion (4 bits, 0-15) |
| 42-47 | age (6 bits, 0-63) |
| 48-63 | staticEval (16-bit signed; INT16_MIN = NO_EVAL sentinel) |

This was validated by SPRT as no-regression (0.490 over 308 games). The optimization was later reverted alongside other changes to isolate a broader regression.

### Known Issues

#### ⚠️ QS In-Check StaticEval Bug (LATENT — Code Reverted)
**The bug**: When the quiescence search (QS) enters **in check**, the `if (!inCheck)` stand-pat block is skipped. The variable `staticEval` stays at 0 (not `NO_EVAL = -32768`). The `storeTT()` was passing this 0 as if it were a valid cached evaluation. This bug was found during investigation but the fix was reverted as part of regression isolation. The baseline code remains with this latent bug.

**The fix** (not currently applied):
```cpp
// Instead of:
storeTT(hash, 0, alpha, qFlag, bestMove, ply, ctx, staticEval);
// Use:
storeTT(hash, 0, alpha, qFlag, bestMove, ply, ctx, haveStaticEval ? staticEval : NO_EVAL);
```

**Status**: SPRT-validated as no-regression (0.480, 153 games) but reverted to match baseline.

#### ⚠️ 0-Node Diagnostic Output (Cosmetic Only)
Search diagnostics print `info string Nodes: 0` regardless of actual node count searched. This is a **pre-existing cosmetic issue** (confirmed by the baseline engine exhibiting the same behavior). It does NOT affect actual play strength.

#### ⚠️ `knowledge.md` is .gitignored
`knowledge.md` is listed in `.gitignore` and is not committed to the repository. It lives on disk for AI agents.

#### ⚠️ Search Parameters Are NNUE-Specific
The search margins (aspiration, RFP, razoring, probcut, delta, null move) are tuned for NNUE's ~3x larger value scale. If you need to run with HCE only, these parameters should be reverted to HCE values.

#### ⚠️ NNUE Auto-Search May Not Cover All Environments
The `readlink("/proc/self/exe", ...)` fallback depends on `/proc` being available (**Linux only**). Won't work on macOS or Windows.

#### ⚠️ INF Reduction May Break Mate Detection in Diagnostics
The `printSearchStats()` function detects mate scores with `abs(score) > INF - 2000`. With INF=60000, mate scores are detected correctly. If `INF` is changed again, the mate detection thresholds must be updated.

---

## 10. TOOLS

### Texel Tuner (`texel_tuner`) — Deep Dive

**Source**: `src/tools/texel_tuner/tuner_main.cpp`
**Build**: `make texel_tuner`
**Purpose**: Optimizes the ~1010 HCE evaluation parameters by minimizing MSE between predicted scores and game results (Texel tuning).

#### Algorithm
- **Optimizer**: Adam (mini-batch gradient descent)
- **Feature extraction**: Perturbation-based — for each parameter, increment by +1, measure the score change, store as a sparse feature
- **Sigmoid model**: `P(win) = 1 / (1 + exp(-C * score))`, where `C = K * ln(10) / 400`
- **K calibration**: Brute-force line search from 0.5 to 3.0 in 0.01 steps on initial base scores
- **Loss**: Mean Squared Error (MSE) between predicted and actual game result (1.0 = win, 0.5 = draw, 0.0 = loss)

#### Hyperparameters
| Parameter | Value | Notes |
|-----------|:-----:|-------|
| Epochs | 50 | Max before early stopping |
| Batch size | 2048 | Positions per mini-batch |
| Base learning rate | 0.35 | Cosine-annealed to 0 |
| Beta1 (Adam) | 0.9 | Momentum decay |
| Beta2 (Adam) | 0.999 | RMSprop decay |
| Epsilon | 1e-8 | Numerical stability |
| L2 lambda | 1e-6 | Weight decay regularization |
| Early stopping patience | 5 | Epochs without Val MSE improvement |

#### Input/Output
- **Input**: EPD file with game results: `fen... 1-0` / `0-1` / `1/2-1/2`
  - Uses only first 4 FEN fields (position, side, castling, en-passant)
  - Optional: `--stockfish-data <file>` for refined training targets (format: `fen|score`)
- **Output**: Directly writes to `src/core/parameters.cpp` — replaces the auto-generated file

#### Important Caveats
1. **Single-threaded**: Feature extraction modifies global `g_Params` and static PST arrays — cannot be parallelized safely
2. **PST params require `recompute_pst()`**: Changing PST or piece values invalidates the cached `mgPst`/`egPst` on the Board, so `board.loadParams()` and `recompute_pst()` must be called after each perturbation
3. **King attack params (KingZoneAttack*, KingDangerScaleMg) are excluded** from optimization — kept as fixed constants
4. **Expected param count**: ~1010 (`EXPECTED_PARAMETER_COUNT = 1010` in `parameters.hpp`)
5. **Default dataset**: `zurichess_quiet.epd` if none provided

#### Usage
```bash
# Basic tuning with default dataset
./texel_tuner zurichess_quiet.epd

# With Stockfish scores as training targets
./texel_tuner zurichess_quiet.epd --stockfish-data stockfish_scores.txt
```

### Stockfish Annotator (`stockfish_annotator`)

**Source**: `src/tools/stockfish_annotator/annotator_main.cpp`
**Build**: `make stockfish_annotator`
**Purpose**: Annotates an EPD file with Stockfish evaluations. Creates training targets for the Texel tuner.

#### Architecture
- Forks Stockfish as a **subprocess** (via `fork()` + `execl()`) — no engine source code dependencies
- Communicates via pipes (`pipe()`, `fdopen`)
- Line-buffered writes to child's stdin
- Reads Stockfish's `info` lines to extract scores

#### CLI Arguments
| Flag | Default | Description |
|------|---------|-------------|
| `--epd <file>` | `zurichess_quiet.epd` | Input EPD file |
| `--sf <path>` | `dev/stockfish` | Path to Stockfish binary |
| `--output <file>` | `stockfish_scores.txt` | Output file |
| `--depth <N>` | 12 | Search depth |
| `--max <N>` | 0 (no limit) | Max positions to annotate |
| `--verbose` | false | Show warnings for failed evaluations |

#### Output Format
```
fen_4_fields|score
```
Example: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - |42`

#### Score Parsing
- `score cp N` → centipawn value `N`
- `score mate N` (N > 0) → `100000 - N` (side to move mates)
- `score mate N` (N < 0) → `-100000 - N` (opponent mates)

#### Usage
```bash
./stockfish_annotator --epd positions.epd --sf /usr/local/bin/stockfish --depth 14 --output sf_scores.txt
```

### Texel Tuning Workflow (End-to-End)
1. Collect EPD positions with game results (e.g., from Zurichess dataset)
2. (Optional) Use `stockfish_annotator` to generate refined score targets
3. Run `texel_tuner` to optimize HCE parameters
4. The tuner writes optimized values directly to `src/core/parameters.cpp`
5. Rebuild the engine: `make chess_app`
6. Test the new parameters: run a match vs baseline with cutechess-cli
7. If improvement is confirmed, commit the new `parameters.cpp`

---

## 11. DEVELOPMENT WORKFLOW

### Common Development Loops

#### 1. Implement → Build → Test (Quick)
```bash
# Edit code, then:
cd build && make chess_app -j$(nproc) 2>&1 | grep -E "error|warning"
# Run a quick sanity check:
echo -e "uci\nisready\nucinewgame\nposition startpos\ngo movetime 2000\nquit" | ./chess_app --uci 2>&1 | grep -E "bestmove|info depth|NNUE"
```

#### 2. Parameter Tuning Cycle
```
1. Modify search params in lawliet.cpp
2. Build: make chess_app
3. Quick benchmark: run a depth-limited search on 2-3 positions
4. If promising: run a 100+ game match vs baseline
5. Analyze PGN for wins/losses/draws
6. Iterate
```

#### 3. NNUE Implementation Debug Cycle
```
1. Check NNUE loads: echo -e "uci\nisready" | ./chess_app --uci | grep NNUE
2. Compare eval: run 'eval' UCI command -> check NNUE vs HCE values
3. For incremental updates: search diagnostics (info string) show incremental vs full recompute counts
4. If values diverge mid-search, the accumulator stack or doMove() has a bug
```

### Debugging Techniques

#### Diagnostic Output
Every completed search prints extensive `info string` statistics. Key metrics to watch:
- `Incremental Updates vs Full Recomputations`: Should heavily favor incremental (~99%)
- `Aspiration Fail High/Low`: Should be rare (single digits). Frequent fails = window too narrow
- `TT Cutoffs`: Should be high (~80-90%) for efficient search
- `QS %`: Should be ~40-55% of total nodes. Higher = more tactical positions
- `Branching Factor`: Lower = more efficient. ~1.5-2.5 is typical
- `FHF (First-Move Cutoffs)`: Should be ~85-90% indicating good move ordering

#### Board Invariant Checks
In debug builds (`-DNDEBUG` not set), `doMove()`/`undoMove()` calls `board.checkInvariants()` which verifies:
- Exactly one king per side
- Occupancy matches piece bitboards exactly
- Color bitboards match piece bitboards exactly
- Turn is ±1, en-passant target is valid

If these fail, the board state is corrupted — usually from a bug in `doMove()` or the NNUE accumulator code.

#### Common Pitfalls
1. **Forgetting to rebuild after parameter changes**: `make` only recompiles changed files, but CMake caches — use `cmake ..` first if adding new files
2. **Missing NNUE weight file**: Engine runs but plays terribly — always check `info string NNUE` output
3. **Stale baseline**: If running HCE matches, ensure the baseline build is the latest stable version at `/home/yuzy/Lawliet_baseline/`
4. **Double `applyMove()`**: The bug that caused the worst damage. NNUE code in `doMove()` must NOT call `board.applyMove()` — that's called separately

### Benchmark Positions

#### Startpos (Standard Starting Position)
```bash
position startpos
go movetime 5000
# Expected (NNUE): ~D14, ~520K NPS, ~64 cp, bestmove d2d4
```

#### Kiwipete (Position 3 of ET-19 suite)
```
position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
go movetime 5000
# Expected (NNUE): ~D8, ~455K NPS, ~-253 cp, bestmove e2a6
```

#### Tactical Test
```
position fen 1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1
go depth 12
# Expected: finds the tactical sequence quickly
```

---

## 12. MULTITHREADING (Lazy SMP)

- `numThreads` controlled by UCI `Threads` option
- Worker threads share the Transposition Table (no locking needed for atomic TTable)
- Each thread has its own `SearchContext` (separate killers, history, NNUE accumulators, pawn table)
- Helper threads diversify search via:
  - **Depth-based reduction**: `if (ctx.threadId > 0) red +/- 1` based on parity
  - Different iteration timing from main thread
- Main thread starts the search; helpers join after signaling

---

## 13. NNUE ARCHITECTURE DETAILS

### Feature Extraction (HalfKP)
- Feature index formula: `PS_END * kingSq + kpp_table[pieceIdx] + pieceSq`
- `kpp_white` maps: white P,N,B,R,Q,K → PS_W_PAWN, etc.; black pieces have different indices
- `kpp_black` maps: different orientation
- Total inputs: `64 * 641 = 41024` (64 king positions × 641 piece-square values)
- Kings are always excluded: `kpp_*[5] = -1` and `kpp_*[11] = -1`

### Forward Pass (SIMD)
```
FT: accumulate 41024 int16 weights into 256-dim accumulators (AVX2)
CReLU: clamp [0, 127], pack int16 → uint8
L1: 512 uint8 → 32 int32 via _mm256_maddubs_epi16 + _mm256_madd_epi16
CReLU: (x >> 6) clamp [0, 127], pack uint8
L2: 32 uint8 → 32 int32 (same SIMD pattern)
CReLU: (x >> 6) clamp [0, 127]
Output: 32 int8 dot product + bias → (result / 16)
```

### Memory Alignment
All network weights are 64-byte aligned (`posix_memalign`) for AVX2 256-bit loads. The accumulator stack uses `alignas(32)` for `__m256i` operations.

---

## 14. CODEBASE CONVENTIONS

### Naming
- `camelCase` for variables and methods
- `PascalCase` for classes
- `UPPER_CASE` for constants (not consistently applied)
- Hungarian-style prefixes not used

### C++ Features Used
- `constexpr` for compile-time constants
- `std::atomic` for lock-free TT access
- `__builtin_ctzll`, `__builtin_popcountll`, `__builtin_bswap16/64` for bit operations
- `_mm256_*` intrinsics for AVX2 SIMD
- `std::thread` for Lazy SMP workers

### Error Handling
- Fatal errors in debug: `std::abort()` via `checkInvariants()`
- File loading failures: print `info string` warnings, return false
- No exceptions used

### Diagnostics
Search produces extensive `info string` diagnostics when search completes:
- Depth, score, NPS, branching factor
- TT hit rates, cutoff rates, occupancy
- Move ordering stats (histograms, first-move cutoff rates)
- Aspiration window fail rates
- LMR statistics
- Root iteration details

---

## 15. DEPENDENCIES

| Dependency | Version | Purpose | Required for |
|------------|---------|---------|-------------|
| SFML 3 | >= 3.0 | Graphics, Window, System | GUI sandbox only |
| GCC/Clang | C++20 | Compiler with AVX2 support | Engine |
| Linux | - | `/proc/self/exe` for NNUE auto-search | Optional fallback (Linux-only) |
| cutechess-cli | - | Match testing | Testing only |

---

## 16. QUICK REFERENCE FOR COMMON TASKS

### Run a Match
```bash
cutechess-cli -tournament gauntlet -rounds 100 -games 2 -repeat -recover \
  -concurrency 4 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -pgnout match.pgn min \
  -engine name="Lawliet-INDEV" cmd=./chess_app proto=uci \
    option.NNUEPath=/path/to/nn-62ef826d1a6d.nnue \
    option.BookPath=/path/to/book.bin \
  -engine name="Lawliet-BASELINE" cmd=/path/to/baseline/chess_app proto=uci \
    option.BookPath=/path/to/book.bin \
  -each tc=10+0.1 option.Threads=1 option.Hash=256
```

### Test / Diagnose NNUE is Working
```bash
# Full diagnostics:
echo -e "uci\nisready\neval\nquit" | ./chess_app --uci 2>&1
```
**Expected successful loading output**:
```
info string NNUE weights loaded from: /path/to/nn-62ef826d1a6d.nnue
info string NNUE evaluation: 57 cp
info string HCE evaluation: 12 cp
```
**Failure output** (engine falls back to HCE with NNUE-scaled params!):
```
info string NNUE weights not found at: nn-62ef826d1a6d.nnue
info string NNUE not available, using hand-crafted evaluation.
info string HCE evaluation: 12 cp
```
If NNUE evaluation is missing, the engine is playing without it — check that the weight file is accessible.

### Quick grep for loading status:
```bash
./chess_app --uci 2>&1 | grep -E "NNUE weights|NNUE not"
# "loaded from" = ✅  "not found" or "not available" = ❌
```

### Build and Run
```bash
cd build && cmake .. && make chess_app -j$(nproc)
./chess_app --uci
```

### Run UCI Commands
```bash
echo -e "uci\nisready\nucinewgame\nposition startpos\ngo depth 12\nquit" | ./chess_app --uci
```

---

## 17. GIT & VERSION HISTORY

### Commit Instructions
After every individual change, run `git commit`. Follow the structure and style of the previous commits (the rather unprofessional commits are only to be made by the user). Additionally, you can use the commit history to find previous changes.

### Baseline Build
Before doing any work, copy or clone the repository as is, then put it in `/home/yuzy/Lawliet_baseline/`. This is the version to use when comparing and validating changes.

### Additional Files in Project Root
- `uci_tuner` — Compiled binary (possibly a UCI-based tuning utility or experimental tool)
- `LICENSE` — GPL v3
- `nn-62ef826d1a6d.nnue` — Stockfish 13 NNUE weight file (the primary evaluation network)

---

*Generated for future AI coding agents. Last updated: July 2026.*
