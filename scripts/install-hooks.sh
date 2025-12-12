#!/bin/bash
# Script to install and enforce pre-commit hooks
# Run this script after cloning the repository

set -e  # Exit on error

echo "🔧 Installing pre-commit hooks for API leak prevention..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

# Install the git hooks
echo "🔗 Installing git hooks..."
pre-commit install

# Install commit-msg hook (optional, for commit message validation)
pre-commit install --hook-type commit-msg

# Install pre-push hook (additional layer of security)
pre-commit install --hook-type pre-push

# Run on all files to check current state
echo "🔍 Running security checks on all files..."
if pre-commit run --all-files; then
    echo "✅ All security checks passed!"
else
    echo "⚠️  Some checks failed. Please review and fix the issues above."
    echo "💡 Tip: Review SECURITY_SETUP.md for guidance on fixing issues."
    exit 1
fi

# Create a baseline for detect-secrets if it doesn't exist
if [ ! -f .secrets.baseline ]; then
    echo "📝 Creating secrets baseline..."
    detect-secrets scan --baseline .secrets.baseline
fi

echo ""
echo "✅ Pre-commit hooks installed successfully!"
echo ""
echo "📋 What happens now:"
echo "  • Every commit will be automatically checked for secrets"
echo "  • Commits with secrets will be BLOCKED"
echo "  • You can run 'pre-commit run --all-files' to check manually"
echo ""
echo "⚠️  Note: Hooks can be bypassed with 'git commit --no-verify'"
echo "   However, CI/CD checks will catch any bypassed commits!"
echo ""
