#include "core/board.hpp"
#include "ui/interface.hpp"
#include "ai/uci.hpp"
#include <iostream>
#include <unistd.h> // For isatty() on Linux/Mac

int main(int argc, char* argv[]) {
    // Check if --uci was explicitly passed
    bool uciMode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--uci") {
            uciMode = true;
            break;
        }
    }

    // If not explicitly set, check if stdin is a terminal.
    // GUIs like Cute Chess pipe input, so isatty(0) will be 0 (false).
    // Running from terminal will be 1 (true).
    if (!uciMode) {
        if (!isatty(STDIN_FILENO)) {
            uciMode = true;
        }
    }

    if (uciMode) {
        UCI uci;
        uci.loop();
    } else {
        Board board;
        Interface ui(board);
        ui.run();
    }
    return 0;
}
