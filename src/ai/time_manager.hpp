#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <cstdint>

class TimeManager {
public:
    mutable std::atomic<bool> stopFlag{false};
    std::atomic<int64_t> nodes{0};
    std::function<void(int depth, int score, int64_t nodes, int timeMs, const std::string& pv)> onInfo;

    std::atomic<int> maxDepthLimit{100};
    std::atomic<int> allocatedTimeMs{0};
    std::atomic<int> totalTimeMs{0};
    std::atomic<bool> infinite{false};
    std::atomic<int64_t> startTimeMs{0}; // Thread-safe atomic start time timestamp

    void startSearch(int wtime, int btime, int winc, int binc, int movestogo, bool isWhite);
    void startDepthSearch(int depth);
    void startMovetimeSearch(int movetime);
    void startInfiniteSearch();

    bool shouldStop() const;
    bool shouldStop(int currentDepth) const;
    bool shouldStopAtRoot(int currentDepth) const;

    void stop();

    int64_t getElapsedMs() const;
};
