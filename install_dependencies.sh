#!/bin/bash
# Smart dependency installer - auto-detects Windows vs Linux

set -e

echo "=============================================="
echo "VeriFile-X Dependency Installer"
echo "=============================================="
echo ""

# Detect platform
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="Windows"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
else
    PLATFORM="Unknown"
fi

echo "Detected platform: $PLATFORM"
echo ""

case $PLATFORM in
    "Windows")
        echo "Installing Windows dependencies..."
        pip install python-magic-bin==0.4.14
        pip install -r backend/requirements.txt
        echo "✓ Windows dependencies installed"
        ;;
    
    "Linux")
        echo "Installing Linux dependencies..."
        echo "Installing system package: libmagic1"
        sudo apt-get update
        sudo apt-get install -y libmagic1 libmagic-dev file
        pip install -r backend/requirements.txt
        echo "✓ Linux dependencies installed"
        ;;
    
    "macOS")
        echo "Installing macOS dependencies..."
        brew install libmagic
        pip install -r backend/requirements.txt
        echo "✓ macOS dependencies installed"
        ;;
    
    *)
        echo "⚠️  Unknown platform, installing base requirements..."
        pip install -r backend/requirements.txt
        ;;
esac

echo ""
echo "Verifying installation..."
python -c "import magic; print('✓ python-magic works')" 2>/dev/null || echo "⚠️  python-magic may need manual setup"
python -c "import psutil; print('✓ psutil works')"
python -c "import fastapi; print('✓ fastapi works')"

echo ""
echo "=============================================="
echo "✓ Installation complete!"
echo "=============================================="
