#!/bin/bash

# Format Python code with black
echo "Formatting Python code with black..."
uv run black backend/

echo "Done! All Python files have been formatted."
