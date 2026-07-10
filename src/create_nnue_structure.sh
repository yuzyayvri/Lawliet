# Create NNUE directory structure
mkdir -p nnue/src/nnue
mkdir -p nnue/src/nnue/architectures
mkdir -p nnue/src/nnue/features
mkdir -p nnue/src/nnue/layers

# Copy all NNUE source files
mkdir -p nnue/src/nnue

# Core NNUE header
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/nnue_common.h -O nnue/src/nnue/nnue_common.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/nnue_architecture.h -O nnue/src/nnue/nnue_architecture.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/nnue_feature_transformer.h -O nnue/src/nnue/nnue_feature_transformer.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/evaluate_nnue.h -O nnue/src/nnue/evaluate_nnue.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/evaluate_nnue.cpp -O nnue/src/nnue/evaluate_nnue.cpp

# Layer files
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/layers/affine_transform.h -O nnue/src/nnue/layers/affine_transform.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/layers/clipped_relu.h -O nnue/src/nnue/layers/clipped_relu.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/layers/input_slice.h -O nnue/src/nnue/layers/input_slice.h

# Feature files
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/features/feature_set.h -O nnue/src/nnue/features/feature_set.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/features/features_common.h -O nnue/src/nnue/features/features_common.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/features/half_kp.cpp -O nnue/src/nnue/features/half_kp.cpp
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/features/half_kp.h -O nnue/src/nnue/features/half_kp.h
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/features/index_list.h -O nnue/src/nnue/features/index_list.h

# Architecture files
wget -q https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_13/src/nnue/architectures/halfkp_256x2-32-32.h -O nnue/src/nnue/architectures/halfkp_256x2-32-32.h

# Create directory structure
mkdir -p src/nnue/src/nnue/nnue
mkdir -p src/nnue/src/nnue/nnue/architectures
mkdir -p src/nnue/src/nnue/nnue/features
mkdir -p src/nnue/src/nnue/nnue/layers

# Move all NNUE files to the correct location
mv nnue/src/nnue/*.h src/nnue/
mv nnue/src/nnue/*.cpp src/nnue/ 2>/dev/null || true

# Create subdirectories
mkdir -p src/nnue/features src/nnue/layers src/nnue/architectures

# Distribute files to subdirectories
mv src/nnue/features.h src/nnue/features/ 2>/dev/null || true
mv src/nnue/features_common.h src/nnue/features/ 2>/dev/null || true
mv src/nnue/halfkp.cpp src/nnue/features/ 2>/dev/null || true
mv src/nnue/halfkp.h src/nnue/features/ 2>/dev/null || true
mv src/nnue/index_list.h src/nnue/features/ 2>/dev/null || true

mv src/nnue/affine_transform.h src/nnue/layers/ 2>/dev/null || true
mv src/nnue/clipped_relu.h src/nnue/layers/ 2>/dev/null || true
mv src/nnue/input_slice.h src/nnue/layers/ 2>/dev/null || true

mv src/nnue/nnue_common.h src/nnue/ 2>/dev/null || true
mv src/nnue/nnue_architecture.h src/nnue/ 2>/dev/null || true
mv src/nnue/nnue_feature_transformer.h src/nnue/ 2>/dev/null || true
mv src/nnue/evaluate_nnue.h src/nnue/ 2>/dev/null || true
mv src/nnue/evaluate_nnue.cpp src/nnue/ 2>/dev/null || true

mv src/nnue/halfkp_256x2-32-32.h src/nnue/architectures/ 2>/dev/null || true

# Create software architecture header
mkdir -p src/ai
