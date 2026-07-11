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
        hardLimitMs.store(5000);
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

    // Hard limit: fixed ceiling that never moves during the search.
    // At most 10% of remaining time, at least 2× allocation.
    int hard = alloc * 2;
    hard = std::min(hard, static_cast<int>(time * 0.10));
    hard = std::max(hard, 100);
    hardLimitMs.store(hard);
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
    hardLimitMs.store(0);
}

void TimeManager::startMovetimeSearch(int movetime) {
    stopFlag.store(false);
    nodes.store(0);
    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    startTimeMs.store(nowMs);
    infinite.store(false);
    allocatedTimeMs.store(movetime);
    totalTimeMs.store(0);  // 0 prevents fail-low/high cap from collapsing allocation
    hardLimitMs.store(std::max(movetime, static_cast<int>(movetime * 1.5)));
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
    hardLimitMs.store(0);
}

bool TimeManager::shouldStop() const {
    if (stopFlag.load()) return true;
    if (infinite.load()) return false;

    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto elapsed = nowMs - startTimeMs.load();
    int alloc = allocatedTimeMs.load();
    int hard = hardLimitMs.load();
    if (elapsed >= alloc || (hard > 0 && elapsed >= hard)) {
        stopFlag.store(true);
        return true;
    }
    return false;
}

bool TimeManager::shouldStop(int currentDepth) const {
    if (currentDepth > maxDepthLimit.load()) {
        stopFlag.store(true);
        return true;
    }
    return shouldStop();
}

bool TimeManager::shouldStopAtRoot(int currentDepth) const {
    if (stopFlag.load()) return true;
    if (currentDepth > maxDepthLimit.load()) return true;
    if (infinite.load()) return false;

    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto elapsed = nowMs - startTimeMs.load();
    int alloc = allocatedTimeMs.load();
    int hard = hardLimitMs.load();

    // Absolute hard limit check (fixed ceiling, never moves)
    if (hard > 0 && elapsed >= hard) {
        stopFlag.store(true);
        return true;
    }        // Prevent premature termination at shallow depths in bullet/blitz time controls
        // In movetime mode (totalTimeMs=0), skip this to use the full requested time.
        if (currentDepth > 6 && elapsed >= alloc * 0.50 && totalTimeMs.load() > 0) {
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
