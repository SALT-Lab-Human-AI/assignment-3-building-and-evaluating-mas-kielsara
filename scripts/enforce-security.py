#!/usr/bin/env python3
"""
Enhanced security enforcement script.
This can be used as a server-side hook or in CI/CD for strict enforcement.
"""

import os
import sys
import subprocess  # nosec B404
import json
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    # Allow shell usage here for pipeline convenience; inputs are static commands
    result = subprocess.run(
        cmd,
        shell=True,  # nosec B602
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        return False
    return result


def check_for_secrets():
    """Run detect-secrets to find hardcoded secrets."""
    print("🔍 Scanning for hardcoded secrets...")

    result = run_command("detect-secrets scan --baseline .secrets.baseline", check=False)

    if result.returncode != 0:
        print("⚠️  Potential secrets detected!")
        print(result.stdout)
        return False

    print("✅ No secrets detected")
    return True


def check_for_api_keys_in_code():
    """Additional check for common API key patterns."""
    print("🔍 Checking for API key patterns...")

    patterns = [
        r"api[_-]?key\s*=\s*['\"][^'\"]{20,}['\"]",
        r"GROQ_API_KEY\s*=\s*['\"]gsk_[^'\"]+['\"]",
        r"OPENAI_API_KEY\s*=\s*['\"]sk-[^'\"]+['\"]",
        r"TAVILY_API_KEY\s*=\s*['\"][^'\"]{30,}['\"]",
    ]

    issues_found = False

    for pattern in patterns:
        result = run_command(
            f"grep -rE '{pattern}' --include='*.py' --include='*.js' --include='*.ts' src/ 2>/dev/null || true",
            check=False
        )
        if result and result.stdout:
            print(f"⚠️  Found potential API key pattern:")
            print(result.stdout)
            issues_found = True

    if not issues_found:
        print("✅ No API key patterns found in code")
        return True

    return False


def check_env_file_not_committed():
    """Ensure .env file is not in git."""
    print("🔍 Checking if .env is properly ignored...")

    result = run_command("git ls-files | grep -E '^\\.env$' || true", check=False)

    if result and result.stdout.strip():
        print("❌ ERROR: .env file is tracked by git!")
        print("   Run: git rm --cached .env")
        return False

    print("✅ .env file is properly ignored")
    return True


def check_large_files():
    """Check for accidentally committed large files."""
    print("🔍 Checking for large files...")

    result = run_command("find . -type f -size +1M | grep -v '.git' || true", check=False)

    if result and result.stdout.strip():
        files = result.stdout.strip().split('\n')
        print(f"⚠️  Found {len(files)} large file(s):")
        for f in files[:5]:  # Show first 5
            print(f"   - {f}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
        return False

    print("✅ No large files found")
    return True


def run_gitleaks():
    """Run gitleaks if available."""
    print("🔍 Running gitleaks scan...")

    # Check if gitleaks is installed
    result = run_command("which gitleaks || true", check=False)
    if not result or not result.stdout.strip():
        print("⚠️  Gitleaks not installed, skipping...")
        return True

    result = run_command("gitleaks detect --no-git -v", check=False)

    if result.returncode != 0:
        print("❌ Gitleaks found potential secrets!")
        print(result.stdout)
        return False

    print("✅ Gitleaks scan passed")
    return True


def main():
    """Run all security checks."""
    print("\n" + "="*60)
    print("🛡️  SECURITY ENFORCEMENT CHECK")
    print("="*60 + "\n")

    checks = [
        ("Secret Detection", check_for_secrets),
        ("API Key Patterns", check_for_api_keys_in_code),
        (".env File Check", check_env_file_not_committed),
        ("Large Files Check", check_large_files),
        ("GitLeaks Scan", run_gitleaks),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        passed = check_func()
        results.append((name, passed))
        print()

    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("="*60 + "\n")

    if all_passed:
        print("✅ All security checks passed!")
        return 0
    else:
        print("❌ Some security checks failed!")
        print("⚠️  Please fix the issues before committing/pushing.")
        print("📖 See SECURITY_SETUP.md for guidance.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
