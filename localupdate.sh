#!/bin/sh
# localupdate.sh — Version v0.0.1
# Rebuilds and updates the globally installed openz binary when you make code changes.

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

echo "🔄 Updating openz v0.0.1 locally..."

# Rebuild release binary
echo "📦 Rebuilding release binary..."
cargo build --release --bin openz

# Check if build succeeded
if [ ! -f target/release/openz ]; then
  die "Build failed! Could not find target/release/openz"
fi

BIN_DIR="$HOME/.cargo/bin"
if [ ! -d "$BIN_DIR" ]; then
  die "$BIN_DIR does not exist. Please run localinstall.sh first."
fi

# Copy binary
echo "🚚 Updating binary in $BIN_DIR/openz..."
cp -f target/release/openz "$BIN_DIR/openz"
chmod +x "$BIN_DIR/openz"

info "openz updated successfully to version v0.0.1!"
