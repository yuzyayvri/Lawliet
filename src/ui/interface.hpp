#pragma once
#include <SFML/Graphics.hpp>
#include <future>
#include <map>
#include <optional>
#include <utility>
#include <vector>
#include "../core/board.hpp"
#include "../ai/lawliet.hpp"

class Interface {
private:
    sf::RenderWindow window;
    sf::View gameView;

    // SFML 3 Compliant Asset Containers
    std::map<int, sf::Texture> pieceTextures;
    std::map<int, sf::Sprite> pieceSprites;

    Board& board;
    Lawliet ai;
    int aiColor = Board::BLACK; // Can be toggled to Board::WHITE

    // Window layout scaling
    const int tileSize = 80;
    const int panelWidth = 200;

    // Drag-and-Drop Interaction states
    bool isDragging = false;
    int selectedSquare = -1;
    int draggedPiece = 0;

    // UI Panel Interactive Controls
    sf::RectangleShape resetBtn;
    sf::RectangleShape prevBtn;
    sf::RectangleShape switchColorBtn;
    sf::Font uiFont;
    sf::Text resetLabel;
    sf::Text prevLabel;
    sf::Text switchColorLabel;

    // Headings, Labels, and Panels
    sf::Text turnText;
    sf::Text historyTitle;
    sf::Text moveHistoryText;
    sf::RectangleShape historyBackground;
    sf::RenderTexture historyRenderTexture;
    std::optional<sf::Sprite> historySprite;
    bool historyTextureReady = false;
    float historyScrollOffset = 0.f;
    float historyMaxScroll = 0.f;

    // Async AI state
    std::optional<std::future<std::pair<uint64_t, Move>>> aiFuture;
    bool aiThinking = false;
    uint64_t aiRequestId = 0;

    // Move Indicator UI Assets
    sf::RectangleShape selectedSquareHighlight;
    std::vector<sf::CircleShape> legalMoveDots;
    std::vector<int> currentLegalMoves;

    // Promotion Picker Assets & Configurations
    bool showingPromoPicker = false;
    int promoFrom = -1;
    int promoTo = -1;
    sf::RectangleShape promoOverlay;
    sf::RectangleShape promoBtns[4];
    const int promoChoices[4] = {5, 2, 4, 3}; // Q, B, R, N internal logic mappings

    // Window System Initializers & Utilities
    void loadAssets();
    void loadUIFont();
    void centerTextOnButton(sf::Text& text, const sf::RectangleShape& btn);
    void clearDrag();
    void updateView();
    sf::Vector2f mapMousePos(sf::Vector2i pixelPos) const;
    void handleEvents();
    void render();

    // UI Render Layers
    void drawBoard();
    void drawPieces();
    void drawSidePanel();
    void drawMoveHistoryPanel();
    void drawPromotionPicker();

    // Game Core Overlays
    void setupPromotionPicker();
    void beginPromotionPicker(int from, int to);
    void cancelPromotionPicker();
    bool handlePromotionClick(sf::Vector2f mousePos);

    // Sidebar Content Refreshers
    void updateTurnDisplay();
    void updateHistoryPanel();
    void updateSwitchColorLabel();
    void undoToPlayerTurn();
    void startAiThink();
    void pollAiThink();
    void cancelAiThink();

    bool isInHistoryPanel(sf::Vector2f pos) const;
    float historyPanelTop() const;
    float historyPanelBottom() const;
    float historyPanelLeft() const;
    float historyPanelWidth() const;
    void initHistoryTexture();
    void refreshHistoryTexture();

    // Move indicator calculations
    void calculateLegalMovesForSquare(int square);

public:
    Interface(Board& b);
    void run();
};
