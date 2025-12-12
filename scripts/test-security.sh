#!/bin/bash
# Test script to verify pre-commit hooks are working correctly

echo "🧪 Testing Pre-commit Security Hooks"
echo "======================================"
echo ""

# Create a temporary test file with a fake API key
TEST_FILE="test_security_check.py"
echo "Creating test file with fake API key..."

cat > $TEST_FILE << 'EOF'
# This is a test file to verify security hooks

# This should be caught by detect-secrets (deliberate fake values for testing)
# pragma: allowlist secret
GROQ_API_KEY = "FAKE_GROQ_KEY_FOR_TESTS"
# pragma: allowlist secret
OPENAI_API_KEY = "FAKE_OPENAI_KEY_FOR_TESTS"

def get_api_key():
    # Bad practice - hardcoded secret
    return "FAKE_KEY_FOR_TESTING_ONLY"

EOF

echo "✅ Test file created: $TEST_FILE"
echo ""

# Try to add it to git
echo "🔍 Attempting to stage file (should be fine)..."
git add $TEST_FILE 2>/dev/null || echo "Note: Not in a git repo or file already tracked"
echo ""

# Try to run pre-commit on it
echo "🔍 Running pre-commit hooks on test file..."
echo "   (This should FAIL and detect secrets)"
echo ""

if pre-commit run --files $TEST_FILE; then
    echo ""
    echo "⚠️  WARNING: Pre-commit did not detect the secrets!"
    echo "   This might mean:"
    echo "   1. Pre-commit hooks are not installed"
    echo "   2. The hooks are not configured correctly"
    echo ""
    echo "   Run: ./scripts/install-hooks.sh"
else
    echo ""
    echo "✅ SUCCESS: Pre-commit correctly blocked the file with secrets!"
    echo "   The security hooks are working properly."
fi

echo ""
echo "🧹 Cleaning up test file..."
rm -f $TEST_FILE
git reset HEAD $TEST_FILE 2>/dev/null

echo ""
echo "======================================"
echo "Test complete!"
echo ""
echo "💡 To install hooks: ./scripts/install-hooks.sh"
echo "📖 Full docs: SECURITY_SETUP.md"
