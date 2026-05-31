#!/bin/sh
# localinstall.sh — Version v0.0.4
# Builds and installs openz locally on this machine so it can be run anywhere.

set -eu

# Terminal colors
if [ -t 1 ]; then
  BOLD='\033[1m' GREEN='\033[32m' YELLOW='\033[33m' RED='\033[31m' RESET='\033[0m'
else
  BOLD='' GREEN='' YELLOW='' RED='' RESET=''
fi

info()  { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn()  { printf "  ${YELLOW}⚠${RESET} %s\n" "$*" >&2; }
die()   { printf "  ${RED}✗${RESET} %s\n" "$*" >&2; exit 1; }
bold()  { printf "${BOLD}%s${RESET}" "$*"; }

echo "🔨 Installing openz v0.0.4 locally..."

# Check rust/cargo
if ! command -v cargo >/dev/null 2>&1; then
  die "Rust/Cargo is not installed. Please install it first from https://rustup.rs"
fi

# Build release binary
echo "📦 Building release binary..."
cargo build --release --bin openz

# Check if build succeeded
if [ ! -f target/release/openz ]; then
  die "Build failed! Could not find target/release/openz"
fi

# Create target bin directory if not exists
BIN_DIR="$HOME/.cargo/bin"
mkdir -p "$BIN_DIR"

# Install binary
echo "🚚 Installing binary to $BIN_DIR/openz..."
cp -f target/release/openz "$BIN_DIR/openz"
chmod +x "$BIN_DIR/openz"

info "openz installed successfully at $BIN_DIR/openz"

# Verify PATH
if echo "$PATH" | grep -q "$BIN_DIR"; then
  info "openz is in your PATH. You can run it using: openz"
else
  warn "$BIN_DIR is not in your PATH."
  echo "Please add the following line to your shell configuration file (e.g., ~/.bashrc or ~/.zshrc):"
  echo "  export PATH=\"\$PATH:\$HOME/.cargo/bin\""
  echo "Then restart your shell or run: source ~/.bashrc"
fi
