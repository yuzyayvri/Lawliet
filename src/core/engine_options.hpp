#pragma once
#include <string>
#include <cstdint>

struct EngineOptions {
    int threads = 1;
    int hashMB = 128;
    std::string bookPath = "book.bin";
    std::string nnuePath = "nnue.bin";
    int moveOverhead = 10;
    bool ponder = false;
};
