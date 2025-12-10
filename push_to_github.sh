#!/bin/bash
# Script to push forensic reconstruction to GitHub
# Usage: ./push_to_github.sh <your-github-username> <repo-name>

set -e  # Exit on error

if [ $# -lt 2 ]; then
    echo "Usage: $0 <github-username> <repo-name>"
    echo "Example: $0 85-Astatine forensic-reconstruction"
    exit 1
fi

GITHUB_USER=$1
REPO_NAME=$2
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "🚀 Preparing to push to GitHub..."
echo "Repository: ${REPO_URL}"
echo ""

# Check if we're in a git repo
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository. Run 'git init' first."
    exit 1
fi

# Stage all changes
echo "📦 Staging files..."
git add .

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short

# Ask for confirmation
echo ""
read -p "Continue with commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Commit
echo ""
echo "💾 Committing changes..."
git commit -m "Add forensic reconstruction system

- Complete text-to-face generation pipeline (Stable Diffusion)
- Face reconstruction pipeline (U-Net with attention)
- Multi-face database with CLIP embeddings
- Comprehensive documentation and analysis"

# Update remote
echo ""
echo "🔗 Updating remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "${REPO_URL}"

# Push
echo ""
echo "⬆️  Pushing to GitHub..."
echo "⚠️  Note: You may need to authenticate with a Personal Access Token"
echo ""
git push -u origin main

echo ""
echo "✅ Success! Your code is now on GitHub:"
echo "   ${REPO_URL}"
echo ""
echo "📝 Next steps:"
echo "   1. Visit your repository on GitHub"
echo "   2. Add a description and topics"
echo "   3. Consider adding a LICENSE file"

