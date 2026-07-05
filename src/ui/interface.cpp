#include "interface.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <chrono>

Interface::Interface(Board& b)
: board(b),
ai(15), // Depth configuration matching lawliet specs
aiColor(Board::BLACK),
resetLabel(uiFont, "Reset Board", 16),
prevLabel(uiFont, "Prev Move", 16),
switchColorLabel(uiFont, "Engine: Black", 16),
    historyTitle(uiFont, "Move History", 14),
    moveHistoryText(uiFont, "", 12),
    turnText(uiFont, "White's Turn", 16) {
        const unsigned int windowWidth = static_cast<unsigned int>(tileSize * 8 + panelWidth);
        const unsigned int windowHeight = static_cast<unsigned int>(tileSize * 8);
        window.create(
            sf::VideoMode({windowWidth, windowHeight}),
                      "Custom Chess Sandbox",
                      sf::Style::Titlebar | sf::Style::Close
        );
        updateView();
        loadAssets();
        loadUIFont();

        resetBtn.setSize({160.f, 40.f});
        resetBtn.setFillColor(sf::Color(70, 70, 70));
        resetBtn.setPosition({static_cast<float>(tileSize * 8 + 20), 40.f});

        prevBtn.setSize({160.f, 40.f});
        prevBtn.setFillColor(sf::Color(70, 70, 70));
        prevBtn.setPosition({static_cast<float>(tileSize * 8 + 20), 100.f});

        switchColorBtn.setSize({160.f, 40.f});
        switchColorBtn.setFillColor(sf::Color(70, 70, 70));
        switchColorBtn.setPosition({static_cast<float>(tileSize * 8 + 20), 160.f});

        resetLabel.setFont(uiFont);
        resetLabel.setString("Reset Board");
        resetLabel.setCharacterSize(16);
        resetLabel.setFillColor(sf::Color::White);
        centerTextOnButton(resetLabel, resetBtn);

        prevLabel.setFont(uiFont);
        prevLabel.setString("Prev Move");
        prevLabel.setCharacterSize(16);
        prevLabel.setFillColor(sf::Color::White);
        centerTextOnButton(prevLabel, prevBtn);

        switchColorLabel.setFont(uiFont);
        switchColorLabel.setString("Engine: Black");
        switchColorLabel.setCharacterSize(16);
        switchColorLabel.setFillColor(sf::Color::White);
        centerTextOnButton(switchColorLabel, switchColorBtn);

        historyTitle.setFont(uiFont);
        historyTitle.setString("Move History");
        historyTitle.setCharacterSize(14);
        historyTitle.setFillColor(sf::Color(180, 180, 180));
        historyTitle.setPosition({static_cast<float>(tileSize * 8 + 20), 220.f});

        moveHistoryText.setFont(uiFont);
        moveHistoryText.setString("");
        moveHistoryText.setCharacterSize(12);
        moveHistoryText.setFillColor(sf::Color::White);

        initHistoryTexture();

        turnText.setFont(uiFont);
        turnText.setCharacterSize(16);
        turnText.setFillColor(sf::Color::Green);

        selectedSquareHighlight.setFillColor(sf::Color(255, 255, 0, 80));
        selectedSquareHighlight.setSize({static_cast<float>(tileSize), static_cast<float>(tileSize)});

        setupPromotionPicker();
        updateTurnDisplay();
        updateHistoryPanel();
    }

    void Interface::calculateLegalMovesForSquare(int square) {
        legalMoveDots.clear();

        int piece = board.getPiece(square);
        if (piece == 0 || ((piece > 0) != (board.turn == Board::WHITE))) {
            return;
        }

        for (int to = 0; to < 64; ++to) {
            if (square == to) continue;

            if (board.isColorMatch(to, board.turn) && !board.isCastlingMove(square, to)) {
                continue;
            }

            int promoChoice = board.isPawnPromotion(square, to) ? 5 : 0; // Using 5 (Queen) for validation mapping

            if (board.leavesKingInCheck(square, to, promoChoice)) {
                sf::CircleShape dot(8.f);
                dot.setFillColor(sf::Color(0, 255, 0, 150));
                int rank = to / 8;
                int file = to % 8;
                dot.setPosition({
                    static_cast<float>(file * tileSize + (tileSize - 16) / 2),
                                static_cast<float>(rank * tileSize + (tileSize - 16) / 2)
                });
                legalMoveDots.push_back(dot);
            }
        }
    }

    void Interface::setupPromotionPicker() {
        promoOverlay.setSize({static_cast<float>(tileSize * 8), static_cast<float>(tileSize * 8)});
        promoOverlay.setPosition({0.f, 0.f});
        promoOverlay.setFillColor(sf::Color(0, 0, 0, 160));
        const float btnSize = static_cast<float>(tileSize) * 0.85f;
        const float totalWidth = btnSize * 4.f + 30.f;
        const float startX = (tileSize * 8.f - totalWidth) / 2.f + 15.f;
        const float y = tileSize * 4.f - btnSize / 2.f;
        for (int i = 0; i < 4; ++i) {
            promoBtns[i].setSize({btnSize, btnSize});
            promoBtns[i].setPosition({startX + i * (btnSize + 10.f), y});
            promoBtns[i].setFillColor(sf::Color(50, 50, 50, 230));
            promoBtns[i].setOutlineThickness(2.f);
            promoBtns[i].setOutlineColor(sf::Color(200, 200, 200));
        }
    }

    void Interface::beginPromotionPicker(int from, int to) {
        promoFrom = from;
        promoTo = to;
        showingPromoPicker = true;
        clearDrag();
        const float btnSize = static_cast<float>(tileSize) * 0.85f;
        const float totalWidth = btnSize * 4.f + 30.f;
        const float startX = (tileSize * 8.f - totalWidth) / 2.f + 15.f;
        const int rank = to / 8;
        const float y = static_cast<float>(rank * tileSize) + (tileSize - btnSize) / 2.f;
        for (int i = 0; i < 4; ++i) {
            promoBtns[i].setPosition({startX + i * (btnSize + 10.f), y});
        }
    }

    void Interface::cancelPromotionPicker() {
        showingPromoPicker = false;
        promoFrom = promoTo = -1;
        currentLegalMoves.clear();
        legalMoveDots.clear();
    }

    bool Interface::handlePromotionClick(sf::Vector2f mousePos) {
        if (!showingPromoPicker) return false;
        for (int i = 0; i < 4; ++i) {
            if (promoBtns[i].getGlobalBounds().contains(mousePos)) {
                if (board.makeMove(promoFrom, promoTo, promoChoices[i])) {
                    cancelPromotionPicker();
                    updateTurnDisplay();
                    updateHistoryPanel();
                    startAiThink();
                }
                return true;
            }
        }
        return true;
    }

    void Interface::updateTurnDisplay() {
        if (aiThinking) {
            turnText.setString("Lawliet is thinking...");
            turnText.setFillColor(sf::Color(180, 180, 255));
            const sf::FloatRect bounds = turnText.getLocalBounds();
            const float panelLeft = static_cast<float>(tileSize * 8);
            const float panelBottom = static_cast<float>(tileSize * 8);
            constexpr float margin = 12.f;
            turnText.setPosition({
                panelLeft + panelWidth - bounds.size.x - margin,
                panelBottom - bounds.size.y - margin
            });
            return;
        }
        const std::string status = board.getStatusText();
        if (!status.empty()) {
            turnText.setString(status);
            if (board.isGameOver()) {
                turnText.setFillColor(sf::Color(255, 200, 80));
            } else if (board.isInCheck(board.turn)) {
                turnText.setFillColor(sf::Color(255, 80, 80));
            }
        } else {
            const int turnNum = static_cast<int>(board.moveHistory.size()) + 1;
            const char* colorName = board.turn == Board::WHITE ? "White" : "Black";
            turnText.setString("Turn " + std::to_string(turnNum) + ": " + colorName);
            turnText.setFillColor(sf::Color::Green);
        }
        const sf::FloatRect bounds = turnText.getLocalBounds();
        const float panelLeft = static_cast<float>(tileSize * 8);
        const float panelBottom = static_cast<float>(tileSize * 8);
        constexpr float margin = 12.f;
        turnText.setPosition({
            panelLeft + panelWidth - bounds.size.x - margin,
            panelBottom - bounds.size.y - margin
        });
    }

    void Interface::updateHistoryPanel() {
        std::ostringstream oss;
        for (size_t i = 0; i < board.moveNotation.size(); ++i) {
            if (i % 2 == 0) {
                oss << (i / 2 + 1) << ". " << board.moveNotation[i];
            } else {
                oss << "  " << board.moveNotation[i] << '\n';
            }
        }
        if (!board.moveNotation.empty() && board.moveNotation.size() % 2 == 1) {
            oss << '\n';
        }
        moveHistoryText.setString(oss.str());
        const float visibleHeight = historyPanelBottom() - historyPanelTop();
        const float textHeight = moveHistoryText.getLocalBounds().size.y;
        historyMaxScroll = std::max(0.f, textHeight - visibleHeight);
        historyScrollOffset = historyMaxScroll;
        refreshHistoryTexture();
    }

    void Interface::updateSwitchColorLabel() {
        std::string text = "Engine: " + std::string(aiColor == Board::WHITE ? "White" : "Black");
        switchColorLabel.setString(text);
        centerTextOnButton(switchColorLabel, switchColorBtn);
    }

    float Interface::historyPanelLeft() const {
        return static_cast<float>(tileSize * 8 + 20);
    }

    float Interface::historyPanelWidth() const {
        return static_cast<float>(panelWidth - 40);
    }

    float Interface::historyPanelTop() const {
        return 245.f;
    }

    float Interface::historyPanelBottom() const {
        return static_cast<float>(tileSize * 8) - 50.f;
    }

    void Interface::initHistoryTexture() {
        const sf::Vector2u size(
            static_cast<unsigned>(historyPanelWidth()),
                                static_cast<unsigned>(historyPanelBottom() - historyPanelTop())
        );
        if (!historyTextureReady) {
            if (!historyRenderTexture.resize(size)) {
                std::cerr << "Failed to resize history render texture\n";
                return;
            }
            historySprite.emplace(historyRenderTexture.getTexture());
            historyTextureReady = true;
        }
        historyBackground.setSize({historyPanelWidth(), historyPanelBottom() - historyPanelTop()});
        historyBackground.setPosition({historyPanelLeft(), historyPanelTop()});
        historyBackground.setFillColor(sf::Color(45, 45, 45));
        if (historySprite) {
            historySprite->setPosition({historyPanelLeft(), historyPanelTop()});
        }
    }

    void Interface::refreshHistoryTexture() {
        if (!historyTextureReady) initHistoryTexture();
        historyRenderTexture.clear(sf::Color(45, 45, 45));
        const sf::FloatRect bounds = moveHistoryText.getLocalBounds();
        moveHistoryText.setPosition({0.f, -historyScrollOffset - bounds.position.y});
        historyRenderTexture.draw(moveHistoryText);
        historyRenderTexture.display();
    }

    bool Interface::isInHistoryPanel(sf::Vector2f pos) const {
        const sf::FloatRect bounds = historyBackground.getGlobalBounds();
        return bounds.contains(pos);
    }

    void Interface::cancelAiThink() {
        ++aiRequestId;
        if (aiFuture) {
            if (aiFuture->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                aiFuture->get();
            }
            aiFuture.reset();
        }
        aiThinking = false;
    }

    void Interface::startAiThink() {
        if (aiThinking || board.isGameOver() || board.turn != aiColor) return;
        aiThinking = true;
        const uint64_t requestId = ++aiRequestId;
        Board boardSnapshot = board;
        aiFuture = std::async(std::launch::async, [this, boardSnapshot, requestId]() mutable {
            Move move = ai.think(boardSnapshot);
            return std::pair{requestId, move};
        });
    }

    void Interface::pollAiThink() {
        if (!aiFuture) return;
        if (aiFuture->wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        const auto [requestId, aiMove] = aiFuture->get();
        aiFuture.reset();
        aiThinking = false;
        if (requestId != aiRequestId) return;
        if (board.isGameOver() || board.turn != aiColor) return;
        int promotionChoiceValue = 0;
        if (aiMove.promotionPiece != 0) {
            promotionChoiceValue = std::abs(aiMove.promotionPiece);
        }
        if (board.makeMove(aiMove.fromSquare, aiMove.toSquare, promotionChoiceValue)) {
            updateTurnDisplay();
            updateHistoryPanel();
        }
    }

    void Interface::undoToPlayerTurn() {
        if (board.moveHistory.empty()) return;
        board.undoMove();
        while (!board.moveHistory.empty() && board.turn == aiColor) {
            board.undoMove();
        }
    }

    void Interface::loadUIFont() {
        const char* fontPaths[] = {
            "assets/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        };
        for (const char* path : fontPaths) {
            if (uiFont.openFromFile(path)) return;
        }
        std::cerr << "Failed to load UI font\n";
    }

    void Interface::centerTextOnButton(sf::Text& text, const sf::RectangleShape& btn) {
        const sf::FloatRect textBounds = text.getLocalBounds();
        text.setOrigin({
            textBounds.position.x + textBounds.size.x / 2.f,
            textBounds.position.y + textBounds.size.y / 2.f
        });
        const sf::FloatRect btnBounds = btn.getGlobalBounds();
        text.setPosition({
            btnBounds.position.x + btnBounds.size.x / 2.f,
            btnBounds.position.y + btnBounds.size.y / 2.f
        });
    }

    void Interface::clearDrag() {
        isDragging = false;
        selectedSquare = -1;
        draggedPiece = 0;
        currentLegalMoves.clear();
        legalMoveDots.clear();
    }

    void Interface::loadAssets() {
        std::string whiteNames[] = {"wP", "wN", "wB", "wR", "wQ", "wK"};
        std::string blackNames[] = {"bP", "bN", "bB", "bR", "bQ", "bK"};
        for (int i = 1; i <= 6; ++i) {
            int wID = i;
            std::string wPath = "assets/" + whiteNames[i - 1] + ".png";
            if (!pieceTextures[wID].loadFromFile(wPath)) {
                std::cerr << "Failed to load: " << wPath << "\n";
            } else {
                pieceSprites.emplace(wID, pieceTextures[wID]);
                auto imgSize = pieceTextures[wID].getSize();
                pieceSprites.at(wID).setScale({static_cast<float>(tileSize) / imgSize.x, static_cast<float>(tileSize) / imgSize.y});
            }
            int bID = -i;
            std::string bPath = "assets/" + blackNames[i - 1] + ".png";
            if (!pieceTextures[bID].loadFromFile(bPath)) {
                std::cerr << "Failed to load: " << bPath << "\n";
            } else {
                pieceSprites.emplace(bID, pieceTextures[bID]);
                auto imgSize = pieceTextures[bID].getSize();
                pieceSprites.at(bID).setScale({static_cast<float>(tileSize) / imgSize.x, static_cast<float>(tileSize) / imgSize.y});
            }
        }
    }

    void Interface::updateView() {
        const float gameWidth = static_cast<float>(tileSize * 8 + panelWidth);
        const float gameHeight = static_cast<float>(tileSize * 8);
        gameView = sf::View(sf::FloatRect({0.f, 0.f}, {gameWidth, gameHeight}));
        const sf::Vector2u windowSize = window.getSize();
        if (windowSize.x == 0 || windowSize.y == 0) {
            window.setView(gameView);
            return;
        }
        const float windowRatio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
        const float gameRatio = gameWidth / gameHeight;
        if (windowRatio > gameRatio) {
            const float viewportWidth = gameRatio / windowRatio;
            const float viewportX = (1.f - viewportWidth) / 2.f;
            gameView.setViewport(sf::FloatRect({viewportX, 0.f}, {viewportWidth, 1.f}));
        } else {
            const float viewportHeight = windowRatio / gameRatio;
            const float viewportY = (1.f - viewportHeight) / 2.f;
            gameView.setViewport(sf::FloatRect({0.f, viewportY}, {1.f, viewportHeight}));
        }
        window.setView(gameView);
    }

    sf::Vector2f Interface::mapMousePos(sf::Vector2i pixelPos) const {
        return window.mapPixelToCoords(pixelPos, gameView);
    }

    void Interface::run() {
        while (window.isOpen()) {
            handleEvents();
            render();
        }
    }

    void Interface::handleEvents() {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
            if (event->is<sf::Event::Resized>()) {
                updateView();
            }
            if (const auto* wheel = event->getIf<sf::Event::MouseWheelScrolled>()) {
                const sf::Vector2f mousePos = mapMousePos(wheel->position);
                if (isInHistoryPanel(mousePos) && historyMaxScroll > 0.f) {
                    historyScrollOffset -= wheel->delta * 18.f;
                    historyScrollOffset = std::clamp(historyScrollOffset, 0.f, historyMaxScroll);
                    refreshHistoryTexture();
                }
            }
            if (const auto* mousePressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mousePressed->button == sf::Mouse::Button::Left) {
                    const sf::Vector2f mousePos = mapMousePos(mousePressed->position);
                    if (showingPromoPicker) {
                        handlePromotionClick(mousePos);
                        continue;
                    }
                    if (resetBtn.getGlobalBounds().contains(mousePos)) {
                        clearDrag();
                        cancelPromotionPicker();
                        cancelAiThink();
                        board.reset();
                        historyScrollOffset = 0.f;
                        updateTurnDisplay();
                        updateHistoryPanel();
                        if (board.turn == aiColor && !board.isGameOver()) {
                            startAiThink();
                        }
                    }
                    else if (prevBtn.getGlobalBounds().contains(mousePos)) {
                        clearDrag();
                        cancelPromotionPicker();
                        cancelAiThink();
                        undoToPlayerTurn();
                        historyScrollOffset = 0.f;
                        updateTurnDisplay();
                        updateHistoryPanel();
                    }
                    else if (switchColorBtn.getGlobalBounds().contains(mousePos)) {
                        clearDrag();
                        cancelPromotionPicker();
                        cancelAiThink();
                        aiColor = -aiColor;
                        updateSwitchColorLabel();
                        updateTurnDisplay();
                        if (board.turn == aiColor && !board.isGameOver()) {
                            startAiThink();
                        }
                    }
                    else if (!aiThinking && !board.isGameOver() && board.turn != aiColor
                        && mousePos.x >= 0.f && mousePos.x < tileSize * 8
                        && mousePos.y >= 0.f && mousePos.y < tileSize * 8) {
                        const int file = static_cast<int>(mousePos.x) / tileSize;
                    const int rank = static_cast<int>(mousePos.y) / tileSize;
                    const int sq = rank * 8 + file;

                    // FIXED: Use getPiece() instead of squares[]
                    if (sq >= 0 && sq < 64 && board.getPiece(sq) != 0
                        && board.isColorMatch(sq, board.turn)) {
                        isDragging = true;
                    selectedSquare = sq;

                    // FIXED: Use getPiece() instead of squares[]
                    draggedPiece = board.getPiece(sq);
                    calculateLegalMovesForSquare(selectedSquare);
                        }
                        }
                }
            }
            if (const auto* mouseReleased = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseReleased->button == sf::Mouse::Button::Left) {
                    if (isDragging && !showingPromoPicker) {
                        const sf::Vector2f mousePos = mapMousePos(mouseReleased->position);
                        if (mousePos.x >= 0.f && mousePos.x < tileSize * 8
                            && mousePos.y >= 0.f && mousePos.y < tileSize * 8) {
                            const int file = static_cast<int>(mousePos.x) / tileSize;
                        const int rank = static_cast<int>(mousePos.y) / tileSize;
                        const int targetSq = rank * 8 + file;
                        if (targetSq >= 0 && targetSq < 64) {
                            if (board.isLegalPromotionMove(selectedSquare, targetSq)) {
                                beginPromotionPicker(selectedSquare, targetSq);
                                return;
                            } else if (board.makeMove(selectedSquare, targetSq)) {
                                updateTurnDisplay();
                                updateHistoryPanel();
                                startAiThink();
                            }
                        }
                            }
                            isDragging = false;
                            selectedSquare = -1;
                            draggedPiece = 0;
                            currentLegalMoves.clear();
                            legalMoveDots.clear();
                    }
                }
            }
        }
        pollAiThink();
        if (!aiThinking && !board.isGameOver() && board.turn == aiColor) {
            startAiThink();
        }
    }

    void Interface::drawBoard() {
        for (int r = 0; r < 8; ++r) {
            for (int f = 0; f < 8; ++f) {
                sf::RectangleShape cell({static_cast<float>(tileSize), static_cast<float>(tileSize)});
                cell.setPosition({static_cast<float>(f * tileSize), static_cast<float>(r * tileSize)});
                if ((r + f) % 2 == 0) cell.setFillColor(sf::Color(240, 240, 220));
                else cell.setFillColor(sf::Color(120, 140, 100));
                window.draw(cell);
            }
        }
        const int kingSq = board.findKing(board.turn);
        if (kingSq >= 0 && board.isInCheck(board.turn)) {
            const int f = kingSq % 8;
            const int r = kingSq / 8;
            sf::RectangleShape highlight({static_cast<float>(tileSize), static_cast<float>(tileSize)});
            highlight.setPosition({static_cast<float>(f * tileSize), static_cast<float>(r * tileSize)});
            highlight.setFillColor(sf::Color(255, 60, 60, 120));
            window.draw(highlight);
        }
    }

    void Interface::drawPieces() {
        const sf::Vector2f mousePos = mapMousePos(sf::Mouse::getPosition(window));
        for (int i = 0; i < 64; ++i) {
            if (isDragging && i == selectedSquare) continue;

            // FIXED: Use getPiece() instead of squares[]
            int piece = board.getPiece(i);
            if (piece != 0 && pieceSprites.contains(piece)) {
                int rank = i / 8;
                int file = i % 8;
                sf::Sprite& s = pieceSprites.at(piece);
                s.setPosition({static_cast<float>(file * tileSize), static_cast<float>(rank * tileSize)});
                window.draw(s);
            }
        }
        if (isDragging && pieceSprites.contains(draggedPiece)) {
            sf::Sprite& s = pieceSprites.at(draggedPiece);
            s.setPosition({mousePos.x - tileSize / 2.0f, mousePos.y - tileSize / 2.0f});
            window.draw(s);
        }
    }

    void Interface::drawMoveHistoryPanel() {
        window.draw(historyTitle);
        if (historyTextureReady && historySprite) {
            window.draw(*historySprite);
        }
    }

    void Interface::drawSidePanel() {
        sf::RectangleShape panel({static_cast<float>(panelWidth), static_cast<float>(tileSize * 8)});
        panel.setPosition({static_cast<float>(tileSize * 8), 0.f});
        panel.setFillColor(sf::Color(35, 35, 35));
        window.draw(panel);
        window.draw(resetBtn);
        window.draw(prevBtn);
        window.draw(switchColorBtn);
        window.draw(resetLabel);
        window.draw(prevLabel);
        window.draw(switchColorLabel);
        drawMoveHistoryPanel();
        window.draw(turnText);
    }

    void Interface::drawPromotionPicker() {
        if (!showingPromoPicker) return;
        window.draw(promoOverlay);

        // FIXED: Use getPiece() instead of squares[]
        const int color = board.getPiece(promoFrom) > 0 ? Board::WHITE : Board::BLACK;
        for (int i = 0; i < 4; ++i) {
            window.draw(promoBtns[i]);
            const int pieceId = color == Board::WHITE ? promoChoices[i] : -promoChoices[i];
            if (pieceSprites.contains(pieceId)) {
                sf::Sprite s = pieceSprites.at(pieceId);
                const sf::FloatRect btnBounds = promoBtns[i].getGlobalBounds();
                const sf::Vector2f spriteSize = s.getGlobalBounds().size;
                s.setPosition({
                    btnBounds.position.x + (btnBounds.size.x - spriteSize.x) / 2.f,
                              btnBounds.position.y + (btnBounds.size.y - spriteSize.y) / 2.f
                });
                window.draw(s);
            }
        }
    }

    void Interface::render() {
        window.clear();
        window.setView(gameView);
        drawBoard();
        if (selectedSquare != -1) {
            int rank = selectedSquare / 8;
            int file = selectedSquare % 8;
            selectedSquareHighlight.setPosition({static_cast<float>(file * tileSize), static_cast<float>(rank * tileSize)});
            window.draw(selectedSquareHighlight);
        }
        drawPieces();
        for (const auto& dot : legalMoveDots) {
            window.draw(dot);
        }
        drawSidePanel();
        drawPromotionPicker();
        window.display();
    }
