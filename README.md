## Lawliet
Lawliet is a chess engine written in C++. It's a hobby project that I started purely out of interest in chess programming and programming in general.

Lawliet speaks UCI, it's compatible with GUIs like [Cute Chess](https://github.com/cutechess/cutechess), which I highly recommend using if you want to use Lawliet with a visual chessboard.

Lawliet does come with a mini chess sandbox, but it's now an unused artifact from Lawliet's early days before I made it speak UCI and I generally do not recommend using it.

Lawliet's first generation of releases are named **Intuition**. After that, the second generation is to be named **Deduction**, followed by **Mastermind** in the future. A generation is dictated by a major milestone or complete overhaul.

Yes, a majority of this was LLM-generated. Yes, I know how to code.

## Building
If you want to build both the executable and texel tuner:
```bash
mkdir build
cd build
cmake ..
make
```

If you don't want to build the texel tuner, or vice versa, simply specify after make:
```bash
make chess_app
```
```bash
make texel_tuner
```

## Usage
Go to the `build` directory in your terminal. After making the executable, you can use `./chess_app` to launch the mini chess sandbox, and `./chess_app --uci` to launch Lawliet with UCI. Alternatively, if you have a GUI, point it towards the `chess_app` executable. Read your specific GUI's documentation if necessary.

## Thanks to
The GOAT, [Stockfish](https://github.com/official-stockfish/Stockfish), for heavy inspiration, and NNUE. My god, implementing NNUE was hell

## License
Lawliet is under the GNU General Public License (GPL v3).


