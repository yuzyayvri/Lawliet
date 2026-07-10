#include "uci.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

void UCI::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        handleCommand(line);
    }
}

void UCI::handleCommand(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    if (cmd == "uci") {
        std::cout << "id name Lawliet" << std::endl;
        std::cout << "id author Yuzy" << std::endl;
        printOptions();
        std::cout << "uciok" << std::endl;
    } else if (cmd == "isready") {
        // Try loading NNUE if a path is configured but not yet loaded
        if (!engine.hasNNUE() && !options.nnuePath.empty()) {
            engine.loadNNUE(options.nnuePath);
            if (engine.hasNNUE()) {
                std::cout << "info string NNUE enabled from: " << options.nnuePath << std::endl;
            }
        }
        std::cout << "readyok" << std::endl;
    } else if (cmd == "ucinewgame") {
        if (searchThread.joinable()) {
            timeManager.stop();
            searchThread.join();
        }
        board.reset();
        engine.reset();
    } else if (cmd == "position") {
        if (searchThread.joinable()) {
            timeManager.stop();
            searchThread.join();
        }
        position(line);
    } else if (cmd == "go") {
        go(line);
    } else if (cmd == "stop") {
        stop();
    } else if (cmd == "quit") {
        stop();
        if (searchThread.joinable()) {
            searchThread.join();
        }
        exit(0);
    } else if (cmd == "setoption") {
        std::string token;
        iss >> token;
        if (token == "name") {
            std::string name = "";
            iss >> token;
            while (token != "value" && !token.empty()) {
                if (!name.empty()) name += " ";
                name += token;
                iss >> token;
            }
            std::string value = "";
            if (token == "value") {
                std::string part;
                while (iss >> part) {
                    if (!value.empty()) value += " ";
                    value += part;
                }
            }
            setOption(name, value);
        }
    }
}

void UCI::printOptions() {
    std::cout << "option name Threads type spin default 1 min 1 max 1024" << std::endl;
    std::cout << "option name Hash type spin default 128 min 1 max 1048576" << std::endl;
    std::cout << "option name Clear Hash type button" << std::endl;
    std::cout << "option name Ponder type check default false" << std::endl;
    std::cout << "option name BookPath type string default book.bin" << std::endl;
    std::cout << "option name Move Overhead type spin default 10 min 0 max 10000" << std::endl;
    std::cout << "option name NNUEPath type string default <empty>" << std::endl;
}

void UCI::setOption(const std::string& name, const std::string& value) {
    std::string lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

    if (lowerName == "threads") {
        int n = std::stoi(value);
        options.threads = std::max(1, n);
        engine.setThreads(options.threads);
    } else if (lowerName == "hash") {
        int mb = std::stoi(value);
        options.hashMB = std::max(1, mb);
        engine.setHashSize(options.hashMB);
    } else if (lowerName == "clear hash") {
        engine.clearHash();
    } else if (lowerName == "bookpath" || lowerName == "book path") {
        options.bookPath = value;
        engine.loadBook(value);
    } else if (lowerName == "ponder") {
        options.ponder = (value == "true");
    } else if (lowerName == "move overhead") {
        options.moveOverhead = std::max(0, std::stoi(value));
    } else if (lowerName == "nnuepath" || lowerName == "nnue path") {
        options.nnuePath = value;
        if (!value.empty()) {
            if (engine.loadNNUE(value)) {
                std::cout << "info string NNUE enabled from: " << value << std::endl;
            } else {
                std::cout << "info string Warning: Failed to load NNUE from: " << value << std::endl;
            }
        }
        if (!engine.hasNNUE()) {
            std::cout << "info string NNUE not available, using hand-crafted evaluation." << std::endl;
        }
    }
}

void UCI::position(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd, token;
    iss >> cmd;
    iss >> token;

    if (token == "startpos") {
        board.reset();
        iss >> token;
    } else if (token == "fen") {
        std::string fen = "";
        while (iss >> token) {
            if (token == "moves") break;
            fen += token + " ";
        }
        board.loadFen(fen);
    }

    if (token == "moves") {
        while (iss >> token) {
            Move m = stringToMove(token);
            if (m.fromSquare != -1) {
                int promoChoice = m.promotionPiece != 0 ? std::abs(m.promotionPiece) : 0;
                board.makeMove(m.fromSquare, m.toSquare, promoChoice);
            }
        }
    }
}

void UCI::go(const std::string& line) {
    std::istringstream iss(line);
    std::string cmd, token;
    iss >> cmd;
    int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0, depth = 0, movestogo = 0;
    bool infinite = false;
    while (iss >> token) {
        if (token == "wtime") iss >> wtime;
        else if (token == "btime") iss >> btime;
        else if (token == "winc") iss >> winc;
        else if (token == "binc") iss >> binc;
        else if (token == "movetime") iss >> movetime;
        else if (token == "depth") iss >> depth;
        else if (token == "infinite") infinite = true;
        else if (token == "movestogo") iss >> movestogo;
    }
    if (searchThread.joinable()) {
        timeManager.stop();
        searchThread.join();
    }
    if (movetime > 0) {
        engine.setDepth(64);
        timeManager.startMovetimeSearch(movetime);
    } else if (depth > 0) {
        engine.setDepth(depth);
        timeManager.startDepthSearch(depth);
    } else if (infinite) {
        engine.setDepth(64);
        timeManager.startInfiniteSearch();
    } else {
        int wtimeAdj = std::max(0, wtime - options.moveOverhead);
        int btimeAdj = std::max(0, btime - options.moveOverhead);
        timeManager.maxDepthLimit.store(64, std::memory_order_relaxed);
        engine.setDepth(64);
        timeManager.startSearch(wtimeAdj, btimeAdj, winc, binc, movestogo, board.turn == Board::WHITE);
    }
    timeManager.onInfo = [this](int depth, int score, int64_t nodes, int timeMs, const std::string& pv) {
        printInfo(depth, score, nodes, timeMs, pv);
    };
    searchThread = std::thread(&UCI::search, this);
}

void UCI::search() {
    Move bestMove = engine.think(board, timeManager);
    std::cout << "bestmove " << moveToString(bestMove) << std::endl;
}

void UCI::stop() {
    timeManager.stop();
}

std::string UCI::moveToString(const Move& m) {
    std::string str = "";
    str += static_cast<char>('a' + (m.fromSquare % 8));
    str += static_cast<char>('1' + (7 - m.fromSquare / 8));
    str += static_cast<char>('a' + (m.toSquare % 8));
    str += static_cast<char>('1' + (7 - m.toSquare / 8));
    if (m.promotionPiece != 0) {
        int type = std::abs(m.promotionPiece);
        char promo = 'q';
        if (type == 2) promo = 'n';
        else if (type == 3) promo = 'b';
        else if (type == 4) promo = 'r';
        str += promo;
    }
    return str;
}

Move UCI::stringToMove(const std::string& str) {
    Move m{};
    m.fromSquare = -1;
    if (str.length() < 4) return m;

    int fromFile = str[0] - 'a';
    int fromRank = '8' - str[1];
    int toFile = str[2] - 'a';
    int toRank = '8' - str[3];

    m.fromSquare = fromRank * 8 + fromFile;
    m.toSquare = toRank * 8 + toFile;

    if (str.length() > 4) {
        char promo = str[4];
        int type = 0;
        if (promo == 'q') type = 5;
        else if (promo == 'r') type = 4;
        else if (promo == 'b') type = 3;
        else if (promo == 'n') type = 2;

        m.promotionPiece = (board.turn == Board::WHITE) ? type : -type;
    }
    return m;
}

void UCI::printInfo(int depth, int score, int64_t nodes, int timeMs, const std::string& pv) {
    std::cout << "info depth " << depth;
    if (score > 9000000) {
        int mateIn = (10000000 - score + 1) / 2;
        std::cout << " score mate " << mateIn;
    } else if (score < -9000000) {
        int mateIn = -(10000000 + score + 1) / 2;
        std::cout << " score mate " << mateIn;
    } else {
        std::cout << " score cp " << score;
    }
    std::cout << " nodes " << nodes;
    if (timeMs > 0) {
        std::cout << " nps " << (nodes * 1000 / timeMs);
    }
    std::cout << " time " << timeMs;
    if (!pv.empty()) {
        std::cout << " pv " << pv;
    }
    std::cout << std::endl;
}
