#!/bin/bash

# Run all quality checks
echo "Running code quality checks..."
echo ""

# Check formatting with black
echo "=== Checking code formatting with black ==="
uv run black --check backend/
BLACK_EXIT=$?

echo ""

# Run tests
echo "=== Running tests ==="
cd backend && uv run pytest tests/ -v
PYTEST_EXIT=$?
cd ..

echo ""
echo "=== Summary ==="

if [ $BLACK_EXIT -eq 0 ]; then
    echo "Formatting: PASSED"
else
    echo "Formatting: FAILED (run ./format.sh to fix)"
fi

if [ $PYTEST_EXIT -eq 0 ]; then
    echo "Tests: PASSED"
else
    echo "Tests: FAILED"
fi

# Exit with error if any check failed
if [ $BLACK_EXIT -ne 0 ] || [ $PYTEST_EXIT -ne 0 ]; then
    exit 1
fi

echo ""
echo "All checks passed!"
