#!/bin/bash
# Fix SDL2 conflict between OpenCV and Pygame
# This script creates a symlink from OpenCV's SDL2 to pygame's SDL2
# so both libraries use the same SDL2 instance, eliminating duplicate class warnings

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12")
CV2_SDL2_PATH=".venv/lib/python${PYTHON_VERSION}/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib"
PYGAME_SDL2_PATH=".venv/lib/python${PYTHON_VERSION}/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib"
CV2_SDL2_DISABLED="${CV2_SDL2_PATH}.disabled"

# Remove disabled file if it exists
if [ -f "$CV2_SDL2_DISABLED" ]; then
    echo "Removing disabled SDL2 library..."
    rm "$CV2_SDL2_DISABLED"
fi

# Check if pygame's SDL2 exists
if [ ! -f "$PYGAME_SDL2_PATH" ]; then
    echo "⚠ Error: pygame's SDL2 library not found at $PYGAME_SDL2_PATH"
    echo "   Make sure pygame is installed in the virtual environment"
    exit 1
fi

# Create symlink if it doesn't exist or is broken
if [ ! -L "$CV2_SDL2_PATH" ] || [ ! -e "$CV2_SDL2_PATH" ]; then
    echo "Creating symlink from OpenCV's SDL2 to pygame's SDL2..."
    # Remove existing file/symlink if it exists
    rm -f "$CV2_SDL2_PATH"
    # Create symlink (relative path from cv2/.dylibs to pygame/.dylibs)
    ln -sf ../../pygame/.dylibs/libSDL2-2.0.0.dylib "$CV2_SDL2_PATH"
    echo "✓ Symlink created: OpenCV now uses pygame's SDL2 library"
else
    echo "✓ Symlink already exists"
fi
