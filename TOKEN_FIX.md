# 🔧 Token Permission Issue - Fix Guide

## Problem
Your token can **read** the repository but cannot **write** (push) to it. This means the token is missing the `repo` scope.

## Solution: Create New Token with Correct Permissions

### Step 1: Go to Token Settings
1. Visit: **https://github.com/settings/tokens**
2. Find your current token (or delete it if you want to create a new one)
3. Click **"Generate new token"** → **"Generate new token (classic)"**

### Step 2: Set Correct Permissions
**IMPORTANT**: Make sure to check these scopes:
- ✅ **`repo`** (Full control of private repositories)
  - This includes: `repo:status`, `repo_deployment`, `public_repo`, `repo:invite`, `security_events`

### Step 3: Generate and Copy Token
1. Click **"Generate token"**
2. **COPY THE NEW TOKEN** immediately (you won't see it again!)

### Step 4: Update and Push
Once you have the new token, run:

```bash
cd /home/teaching/G14/forensic_reconstruction

# Update remote with new token
git remote set-url origin https://NEW_TOKEN_HERE@github.com/85-Astatine/forensic-reconstruction.git

# Push
git push -u origin main
```

**OR** use the token in the URL format:
```bash
git remote set-url origin https://85-Astatine:NEW_TOKEN_HERE@github.com/85-Astatine/forensic-reconstruction.git
git push -u origin main
```

---

## Alternative: Use SSH (If You Have SSH Keys Set Up)

If you have SSH keys configured on GitHub, you can use SSH instead:

```bash
cd /home/teaching/G14/forensic_reconstruction
git remote set-url origin git@github.com:85-Astatine/forensic-reconstruction.git
git push -u origin main
```

---

## Quick Check: Verify Token Permissions

You can check what permissions your current token has by visiting:
**https://github.com/settings/tokens**

Look for the token and check if `repo` scope is checked.

---

## After Fixing

Once you push successfully, you can:
1. Remove the token from the remote URL for security:
   ```bash
   git remote set-url origin https://github.com/85-Astatine/forensic-reconstruction.git
   ```
2. Use SSH or credential helper for future pushes

