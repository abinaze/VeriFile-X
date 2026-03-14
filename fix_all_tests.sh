#!/bin/bash
set -e

echo "🔧 Fixing VeriFile-X Tests for 21 Signals + v6.0.0"
echo "=================================================="

# Phase 1: Signal Count Updates (19 → 21)
echo "📊 Phase 1: Updating signal counts (19 → 21)..."

# Find all occurrences of signal count assertions
files_to_fix=(
  "backend/tests/test_advanced_ai_detector.py"
  "backend/tests/test_advanced_ensemble.py"
  "backend/tests/test_ai_detector.py"
  "backend/tests/test_covariance_detector.py"
  "backend/tests/test_statistical_detector.py"
  "backend/tests/test_ultra_advanced_detector.py"
)

for file in "${files_to_fix[@]}"; do
  if [ -f "$file" ]; then
    echo "  Fixing: $file"
    
    # Fix total_signals == 19
    sed -i 's/total_signals == 19/total_signals == 21/g' "$file"
    sed -i 's/total_signals"] == 19/total_signals"] == 21/g' "$file"
    sed -i 's/"total_signals": 19/"total_signals": 21/g' "$file"
    sed -i 's/\["total_detection_signals"\] == 19/["total_detection_signals"] == 21/g' "$file"
    sed -i 's/"total_detection_signals"] == 19/"total_detection_signals"] == 21/g' "$file"
    
    # Fix signal count in comments/docstrings
    sed -i 's/19 signals/21 signals/g' "$file"
    sed -i 's/(19 total)/(21 total)/g' "$file"
  fi
done

# Phase 2: Version Updates (5.0.0 → 6.0.0)
echo "📦 Phase 2: Updating analyzer version (5.0.0 → 6.0.0)..."

for file in "${files_to_fix[@]}"; do
  if [ -f "$file" ]; then
    sed -i 's/"5\.0\.0"/"6.0.0"/g' "$file"
    sed -i "s/'5\.0\.0'/'6.0.0'/g" "$file"
  fi
done

# Phase 3: Variance Tolerance Updates
echo "🎲 Phase 3: Updating variance tolerances for CLIP..."

# Fix determinism test - increase tolerance for CLIP randomness
if [ -f "backend/tests/test_determinism.py" ]; then
  echo "  Fixing: backend/tests/test_determinism.py"
  # Will be fixed manually below
fi

echo "✅ Automated fixes complete!"
echo ""
echo "Next: Manual review and testing"
