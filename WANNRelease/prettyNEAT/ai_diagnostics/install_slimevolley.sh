#!/bin/bash

# SlimeVolley Environment Installation Script
# This script installs the required dependencies for SlimeVolley integration

echo "============================================================"
echo "SlimeVolley NEAT Integration - Dependency Installation"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check Python version
echo "1. Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
print_info "Python version: $python_version"

# Check if python version is 3.7+
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)" 2>/dev/null; then
    print_success "Python version is compatible (3.7+)"
else
    print_error "Python 3.7+ required. Current version: $python_version"
    exit 1
fi

echo ""

# Install gymnasium
echo "2. Installing/upgrading gymnasium..."
pip install --upgrade gymnasium > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "gymnasium installed/upgraded"
else
    print_error "Failed to install gymnasium"
    exit 1
fi

echo ""

# Install slimevolleygym
echo "3. Installing slimevolleygym..."
pip install slimevolleygym > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "slimevolleygym installed"
else
    print_error "Failed to install slimevolleygym"
    print_info "Trying with --upgrade flag..."
    pip install --upgrade slimevolleygym
    if [ $? -eq 0 ]; then
        print_success "slimevolleygym installed"
    else
        print_error "Failed to install slimevolleygym"
        exit 1
    fi
fi

echo ""

# Check for optional dependencies
echo "4. Checking optional dependencies..."

# Check for MPI (for parallel training)
if python -c "import mpi4py" 2>/dev/null; then
    print_success "mpi4py found (parallel training enabled)"
else
    print_info "mpi4py not found (parallel training disabled)"
    echo "   To enable parallel training, install with:"
    echo "   pip install mpi4py"
fi

# Check for pygame (for rendering)
if python -c "import pygame" 2>/dev/null; then
    print_success "pygame found (rendering enabled)"
else
    print_info "pygame not found (rendering may be limited)"
    echo "   To enable better rendering, install with:"
    echo "   pip install pygame"
fi

echo ""

# Verify installation
echo "5. Verifying installation..."
python -c "import slimevolleygym; import gymnasium; print('All imports successful')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "All core dependencies verified"
else
    print_error "Verification failed"
    exit 1
fi

echo ""
echo "============================================================"
print_success "Installation complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run tests:    python test_slimevolley.py"
echo "  2. Start training: python neat_train.py -p p/slimevolley_quick.json -n 4"
echo "  3. Read docs:    cat SLIMEVOLLEY_README.md"
echo ""
echo "For more information, see SLIMEVOLLEY_SETUP.md"
echo "============================================================"
