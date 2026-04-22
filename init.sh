#!/usr/bin/env bash
# init.sh — One-time project environment setup
# Usage: bash init.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="flash_flood"

echo "=== Singapore Flash Flood Prediction — Environment Setup ==="
echo ""

# 1. Create conda environment
echo "[1/5] Creating conda environment: $ENV_NAME (Python 3.11)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "      Environment already exists, skipping creation."
else
    conda create -y -n "$ENV_NAME" python=3.11
fi

# 2. Install dependencies
echo "[2/5] Installing Python packages..."
conda run -n "$ENV_NAME" pip install -r "$PROJECT_DIR/requirements.txt"

# 3. Register Jupyter kernel
echo "[3/5] Registering Jupyter kernel..."
conda run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python (flash_flood)"

# 4. Ensure .env exists
echo "[4/5] Checking .env file..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo ""
    echo "  *** ACTION REQUIRED ***"
    echo "  .env created from .env.example."
    echo "  Open .env and fill in your API keys before running any scripts:"
    echo "    - OPENAI_API_KEY"
    echo "    - TELEGRAM_API_ID and TELEGRAM_API_HASH (from https://my.telegram.org)"
    echo "    - ANTHROPIC_API_KEY (optional)"
    echo ""
else
    echo "      .env already exists."
fi

# 5. Install pre-commit hook to catch accidental key leaks
echo "[5/5] Installing pre-commit key-leak guard..."
HOOK_FILE="$PROJECT_DIR/.git/hooks/pre-commit"
if [ -d "$PROJECT_DIR/.git" ]; then
    cat > "$HOOK_FILE" << 'HOOK'
#!/usr/bin/env bash
# Pre-commit hook: block commits containing API key patterns
if git diff --cached --name-only | xargs grep -lE "sk-[a-zA-Z0-9]{20,}|sk-ant-[a-zA-Z0-9\-]{20,}" 2>/dev/null; then
    echo ""
    echo "ERROR: Possible API key detected in staged files. Commit blocked."
    echo "Remove the key, add to .env, and load via python-dotenv instead."
    exit 1
fi
HOOK
    chmod +x "$HOOK_FILE"
    echo "      Pre-commit hook installed."
else
    echo "      No .git directory found — skipping hook (run inside a git repo for key-leak protection)."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "VS Code Jupyter setup:"
echo "  1. Install extension: ms-toolsai.jupyter"
echo "  2. Open any .ipynb notebook"
echo "  3. Select kernel: 'Python (flash_flood)'"
echo ""
echo "To activate conda env in terminal: conda activate $ENV_NAME"
echo "To run the Streamlit dashboard:    streamlit run app/app.py"
