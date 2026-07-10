cmake_minimum_required(VERSION 3.15)
project(CustomChessEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SFML 3 components MUST be capitalized exactly like this:
find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)

include_directories(src)

# Gather all source files recursively
file(GLOB_RECURSE SOURCES "src/*.cpp")

# --- Target 1: chess_app (Main Engine) ---
set(APP_SOURCES ${SOURCES})
# Remove tuner_main.cpp and annotator_main.cpp from the list of engine sources
list(REMOVE_ITEM APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tools/texel_tuner/tuner_main.cpp")
list(REMOVE_ITEM APP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tools/stockfish_annotator/annotator_main.cpp")

add_executable(chess_app ${APP_SOURCES})
target_link_libraries(chess_app SFML::Graphics SFML::Window SFML::System)

# --- Target 2: texel_tuner (Tuning Tool) ---
set(TUNER_SOURCES ${SOURCES})
# Remove main.cpp and annotator_main.cpp from the list of tuner sources
list(REMOVE_ITEM TUNER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM TUNER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tools/stockfish_annotator/annotator_main.cpp")

add_executable(texel_tuner ${TUNER_SOURCES})
target_link_libraries(texel_tuner SFML::Graphics SFML::Window SFML::System)

# --- Target 3: stockfish_annotator (EPD Annotation Tool) ---
add_executable(stockfish_annotator
    src/tools/stockfish_annotator/annotator_main.cpp
)
# No SFML or engine source dependencies needed -- standalone subprocess-based tool

# Post-build copy asset step for the main engine app
add_custom_command(TARGET chess_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:chess_app>/assets
)

