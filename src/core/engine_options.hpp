#pragma once
#include <string>
#include <cstdint>

struct EngineOptions {
    int threads = 1;
    int hashMB = 128;
    std::string bookPath = "book.bin";
    std::string nnuePath = "nn-62ef826d1a6d.nnue";
    int moveOverhead = 10;
    bool ponder = false;
};
