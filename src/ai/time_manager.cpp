#include "time_manager.hpp"
#include <algorithm>

void TimeManager::startSearch(int wtime, int btime, int winc, int binc, int movestogo, bool isWhite) {
    stopFlag.store(false);
    nodes.store(0);
    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    startTimeMs.store(nowMs);
    infinite.store(false);

    int time = isWhite ? wtime : btime;
    int inc = isWhite ? winc : binc;
    totalTimeMs.store(time);

    if (time <= 0) {
        allocatedTimeMs.store(2000);
        infinite.store(true);
        totalTimeMs.store(0);
        return;
    }

    int divisor = (movestogo > 0) ? std::min(movestogo + 2, 40) : 25;
    int alloc = (time / divisor) + static_cast<int>(inc * 0.8);
    alloc = std::max(5, std::min(alloc, time / 2));

    if (alloc > time - 50) {
        alloc = std::max(10, time - 50);
    }
    if (alloc < 10) alloc = 10;
    allocatedTimeMs.store(alloc);
}

void TimeManager::startDepthSearch(int depth) {
    stopFlag.store(false);
    nodes.store(0);
    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    startTimeMs.store(nowMs);
    infinite.store(true);
    allocatedTimeMs.store(0);
    totalTimeMs.store(0);
    maxDepthLimit.store(depth);
}

void TimeManager::startMovetimeSearch(int movetime) {
    stopFlag.store(false);
    nodes.store(0);
    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    startTimeMs.store(nowMs);
    infinite.store(false);
    allocatedTimeMs.store(movetime);
    totalTimeMs.store(movetime);
}

void TimeManager::startInfiniteSearch() {
    stopFlag.store(false);
    nodes.store(0);
    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    startTimeMs.store(nowMs);
    infinite.store(true);
    allocatedTimeMs.store(0);
    totalTimeMs.store(0);
}

bool TimeManager::shouldStop() const {
    if (stopFlag.load()) return true;
    if (infinite.load()) return false;

    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto elapsed = nowMs - startTimeMs.load();
    if (elapsed >= allocatedTimeMs.load()) {
        stopFlag.store(true);
        return true;
    }
    return false;
}

bool TimeManager::shouldStop(int currentDepth) const {
    if (currentDepth >= maxDepthLimit.load()) {
        stopFlag.store(true);
        return true;
    }
    return shouldStop();
}

bool TimeManager::shouldStopAtRoot(int currentDepth) const {
    if (stopFlag.load()) return true;
    if (currentDepth >= maxDepthLimit.load()) return true;
    if (infinite.load()) return false;

    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto elapsed = nowMs - startTimeMs.load();
    int alloc = allocatedTimeMs.load();

    // Prevent premature termination at shallow depths in bullet/blitz time controls
    if (currentDepth > 6 && elapsed >= alloc * 0.65) {
        stopFlag.store(true);
        return true;
    }

    if (elapsed >= alloc) {
        stopFlag.store(true);
        return true;
    }
    return false;
}

void TimeManager::stop() {
    stopFlag.store(true);
}

int64_t TimeManager::getElapsedMs() const {
    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return nowMs - startTimeMs.load();
}
