#pragma once
#include "../core/board.hpp"
#include "../core/engine_options.hpp"
#include "lawliet.hpp"
#include "time_manager.hpp"
#include <string>
#include <thread>

class UCI {
public:
    void loop();
private:
    Board board;
    Lawliet engine;
    TimeManager timeManager;
    std::thread searchThread;
    EngineOptions options;

    void handleCommand(const std::string& line);
    void position(const std::string& line);
    void go(const std::string& line);
    void stop();
    void search();
    void printOptions();
    void setOption(const std::string& name, const std::string& value);

    std::string moveToString(const Move& m);
    Move stringToMove(const std::string& str);
    void printInfo(int depth, int score, int64_t nodes, int timeMs, const std::string& pv);
};
