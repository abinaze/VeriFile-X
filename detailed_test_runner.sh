#!/bin/bash
# Detailed Test Runner - Runs each test file individually and reports results

echo "=================================================="
echo "VeriFile-X DETAILED TEST ANALYSIS"
echo "=================================================="
echo ""

cd ~/VeriFile-X || exit 1

echo "Running each test file individually..."
echo "This helps identify exactly which tests are failing."
echo ""

TOTAL_FILES=0
PASSED_FILES=0
FAILED_FILES=0

for test_file in backend/tests/test_*.py; do
    filename=$(basename "$test_file")
    ((TOTAL_FILES++))
    
    echo "=================================================="
    echo "Testing: $filename"
    echo "=================================================="
    
    # Run the test and capture result
    if pytest "$test_file" -v --tb=line 2>&1 | tee /tmp/test_output.txt; then
        echo "✓ PASSED: $filename"
        ((PASSED_FILES++))
    else
        echo "✗ FAILED: $filename"
        ((FAILED_FILES++))
        
        # Extract failure details
        echo ""
        echo "--- Failure Details ---"
        grep -A 3 "FAILED\|AssertionError\|Error" /tmp/test_output.txt | head -20
    fi
    
    echo ""
done

echo "=================================================="
echo "SUMMARY"
echo "=================================================="
echo "Total test files: $TOTAL_FILES"
echo "Passed: $PASSED_FILES"
echo "Failed: $FAILED_FILES"
echo "=================================================="

if [ $FAILED_FILES -gt 0 ]; then
    echo ""
    echo "Running full test suite to get detailed failure report..."
    echo ""
    pytest backend/tests/ -v --tb=short -x
fi
