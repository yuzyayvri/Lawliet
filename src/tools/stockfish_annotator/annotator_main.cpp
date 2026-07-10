#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// ============================================================================
// UCI Subprocess Manager — forks Stockfish and communicates over pipes
// ============================================================================
struct StockfishProcess {
    pid_t pid = -1;
    FILE* to_child = nullptr;   // write to child's stdin
    FILE* from_child = nullptr; // read from child's stdout

    ~StockfishProcess() { close(); }

    bool launch(const std::string& path) {
        int stdin_pipe[2];  // parent writes to [1], child reads from [0]
        int stdout_pipe[2]; // child writes to [1], parent reads from [0]

        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
            std::cerr << "Error: Failed to create pipes." << std::endl;
            return false;
        }

        pid = fork();
        if (pid < 0) {
            std::cerr << "Error: fork() failed." << std::endl;
            return false;
        }

        if (pid == 0) {
            // Child process: exec Stockfish
            ::close(stdin_pipe[1]);   // close write end of stdin pipe
            ::close(stdout_pipe[0]);  // close read end of stdout pipe

            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);

            ::close(stdin_pipe[0]);
            ::close(stdout_pipe[1]);

            execl(path.c_str(), "stockfish", nullptr);
            // If exec fails:
            std::cerr << "Error: Failed to exec Stockfish at: " << path << std::endl;
            _exit(1);
        }

        // Parent process
        ::close(stdin_pipe[0]);  // close read end of stdin pipe
        ::close(stdout_pipe[1]); // close write end of stdout pipe

        to_child = fdopen(stdin_pipe[1], "w");
        from_child = fdopen(stdout_pipe[0], "r");

        if (!to_child || !from_child) {
            std::cerr << "Error: fdopen failed." << std::endl;
            close();
            return false;
        }

        setlinebuf(to_child); // line-buffered writes
        return true;
    }

    void send(const std::string& cmd) {
        fprintf(to_child, "%s\n", cmd.c_str());
        fflush(to_child);
    }

    // Read until a line containing the given marker string is found, or timeout.
    // Returns the accumulated output up to and including the marker line.
    std::string readUntil(const std::string& marker, int timeout_ms = 5000) {
        std::string output;
        char line[4096];

        while (fgets(line, sizeof(line), from_child)) {
            output += line;
            if (strstr(line, marker.c_str())) {
                break;
            }
        }

        return output;
    }

    // Send a position and search command, then parse the final evaluation.
    // Returns true if a score was successfully parsed.
    bool evaluatePosition(const std::string& fen, int depth,
                          int& outScore, bool& outIsMate) {
        send("position fen " + fen);
        send("go depth " + std::to_string(depth));

        outScore = 0;
        outIsMate = false;
        bool foundScore = false;

        char line[4096];
        while (fgets(line, sizeof(line), from_child)) {
            // Check for bestmove indicating search completion
            if (strncmp(line, "bestmove", 8) == 0) {
                break;
            }

            // Check for info lines with score
            if (strncmp(line, "info", 4) == 0) {
                const char* scorePtr = strstr(line, "score");
                if (!scorePtr) continue;

                if (strstr(scorePtr, "cp ")) {
                    // Centipawn score: "score cp N"
                    const char* cpPtr = scorePtr + 6; // skip "score "
                    if (strncmp(cpPtr, "cp ", 3) == 0) {
                        outScore = std::atoi(cpPtr + 3);
                        outIsMate = false;
                        foundScore = true;
                    }
                } else if (strstr(scorePtr, "mate ")) {
                    // Mate score: "score mate N"
                    const char* matePtr = scorePtr + 6; // skip "score "
                    if (strncmp(matePtr, "mate ", 5) == 0) {
                        int mateN = std::atoi(matePtr + 5);
                        // Convert mate to a large numeric score:
                        // mate N (N>0): side to move mates in N -> positive
                        // mate N (N<0): opponent mates in N -> negative
                        if (mateN > 0) {
                            outScore = 100000 - mateN;
                        } else {
                            outScore = -100000 - mateN; // mateN negative, so this adds
                        }
                        outIsMate = true;
                        foundScore = true;
                    }
                }
            }
        }

        return foundScore;
    }

    void close() {
        if (pid > 0) {
            send("quit");
            int status;
            // Block until Stockfish actually exits after quit command
            waitpid(pid, &status, 0);
            pid = -1;
        }
        if (to_child) { fclose(to_child); to_child = nullptr; }
        if (from_child) { fclose(from_child); from_child = nullptr; }
    }
};

// ============================================================================
// EPD Parsing — extract FEN (first 4 fields) from an EPD line
// ============================================================================
bool parseEpdFen(const std::string& line, std::string& fen) {
    if (line.empty()) return false;

    std::istringstream iss(line);
    std::string part;
    fen = "";
    int fields = 0;
    while (iss >> part && fields < 4) {
        fen += part + " ";
        fields++;
    }
    return fields == 4;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::string epdFile = "zurichess_quiet.epd";
    std::string sfPath = "dev/stockfish";
    std::string outputFile = "stockfish_scores.txt";
    int searchDepth = 12;
    size_t maxPositions = 0; // 0 = no limit
    bool verbose = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--epd" && i + 1 < argc) {
            epdFile = argv[++i];
        } else if (arg == "--sf" && i + 1 < argc) {
            sfPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            searchDepth = std::stoi(argv[++i]);
        } else if (arg == "--max" && i + 1 < argc) {
            maxPositions = std::stoul(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg[0] != '-') {
            epdFile = arg;
        }
    }

    std::cout << "=== Stockfish EPD Annotator ===" << std::endl;
    std::cout << "EPD file:    " << epdFile << std::endl;
    std::cout << "Stockfish:   " << sfPath << std::endl;
    std::cout << "Output:      " << outputFile << std::endl;
    std::cout << "Depth:       " << searchDepth << std::endl;
    if (maxPositions > 0)
        std::cout << "Max pos:     " << maxPositions << std::endl;

    // Open EPD file
    std::ifstream epd(epdFile);
    if (!epd) {
        std::cerr << "Error: Cannot open EPD file: " << epdFile << std::endl;
        return 1;
    }

    // Open output file
    std::ofstream out(outputFile);
    if (!out) {
        std::cerr << "Error: Cannot create output file: " << outputFile << std::endl;
        return 1;
    }

    // Launch Stockfish
    StockfishProcess sf;
    if (!sf.launch(sfPath)) {
        std::cerr << "Error: Failed to launch Stockfish at: " << sfPath << std::endl;
        return 1;
    }

    // Initialize Stockfish (UCI handshake)
    sf.send("uci");
    sf.readUntil("uciok", 10000);
    sf.send("isready");
    sf.readUntil("readyok", 10000);

    std::cout << "Stockfish initialized. Starting annotation..." << std::endl;

    // Read all EPD lines first so we can show progress
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(epd, line)) {
        if (line.empty()) continue;

        std::string fen;
        if (parseEpdFen(line, fen)) {
            lines.push_back(fen);
            if (maxPositions > 0 && lines.size() >= maxPositions)
                break;
        }
    }
    epd.close();

    std::cout << "Loaded " << lines.size() << " positions from EPD." << std::endl;

    // Annotate each position
    size_t annotated = 0;
    size_t failed = 0;

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string& fen = lines[i];

        int score = 0;
        bool isMate = false;
        if (sf.evaluatePosition(fen, searchDepth, score, isMate)) {
            // Write: fen|score
            // Trim trailing space from fen
            std::string fenTrimmed = fen;
            while (!fenTrimmed.empty() && fenTrimmed.back() == ' ')
                fenTrimmed.pop_back();
            out << fenTrimmed << "|" << score << "\n";
            annotated++;
        } else {
            failed++;
            if (verbose) {
                std::cerr << "Warning: Failed to evaluate position " << i << ": " << fen << std::endl;
            }
        }

        if ((i + 1) % 10000 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << lines.size()
                      << " positions processed." << std::endl;
        }
    }

    sf.close();
    out.close();

    std::cout << "Annotation complete." << std::endl;
    std::cout << "  Total positions: " << lines.size() << std::endl;
    std::cout << "  Annotated:       " << annotated << std::endl;
    std::cout << "  Failed:          " << failed << std::endl;
    std::cout << "Output written to: " << outputFile << std::endl;

    return failed > 0 ? 1 : 0;
}
