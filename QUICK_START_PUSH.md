# 🚀 Quick Start: Push to GitHub

## What You Need

1. **GitHub Account**: You already have one (85-Astatine) ✅
2. **Repository Name**: Choose one (e.g., `forensic-reconstruction`)
3. **Personal Access Token**: For authentication (see below)

---

## ⚡ Fastest Way (3 Steps)

### Step 1: Create Repository on GitHub

1. Go to: **https://github.com/new**
2. **Repository name**: `forensic-reconstruction`
3. **Description**: "AI-powered forensic facial reconstruction system"
4. **Visibility**: Choose Public or Private
5. **DO NOT** check "Initialize with README" (we already have one)
6. Click **"Create repository"**

### Step 2: Get Personal Access Token

1. Go to: **https://github.com/settings/tokens**
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. **Note**: "Forensic Reconstruction"
4. **Expiration**: Choose 90 days or No expiration
5. **Scopes**: Check ✅ **`repo`** (this gives full repository access)
6. Click **"Generate token"**
7. **COPY THE TOKEN** (you won't see it again! Save it somewhere safe)

### Step 3: Push to GitHub

**Option A: Use the helper script** (Easiest)
```bash
cd /home/teaching/G14/forensic_reconstruction
./push_to_github.sh 85-Astatine forensic-reconstruction
```
When asked for password, paste your **Personal Access Token** (not your GitHub password!)

**Option B: Manual commands**
```bash
cd /home/teaching/G14/forensic_reconstruction

# Stage all files
git add .

# Commit
git commit -m "Add forensic reconstruction system with Stable Diffusion and U-Net"

# Update remote (replace with YOUR repo name)
git remote set-url origin https://github.com/85-Astatine/forensic-reconstruction.git

# Push
git push -u origin main
```
When asked for password, paste your **Personal Access Token**.

---

## ✅ That's It!

After pushing, visit: **https://github.com/85-Astatine/forensic-reconstruction**

---

## 🆘 Troubleshooting

### "Repository not found"
- Make sure you created the repo on GitHub first (Step 1)
- Check the repository name matches exactly

### "Authentication failed"
- Make sure you're using a **Personal Access Token**, not your password
- Token must have `repo` scope checked

### "Branch 'main' does not exist"
- Your branch is already `main`, so this shouldn't happen
- If it does, run: `git branch -M main`

### "Permission denied"
- Check your Personal Access Token has `repo` scope
- Make sure the repository name is correct

---

## 📋 What Will Be Pushed?

✅ **Included:**
- All source code (`src/`)
- Documentation (README, guides, etc.)
- Configuration files
- Scripts

❌ **Excluded** (too large or sensitive):
- `venv/` (virtual environment)
- `models/` (large AI models)
- `output/` (generated results)
- `logs/` (log files)

---

## 🎯 After Pushing

1. Add repository description on GitHub
2. Add topics: `forensic`, `face-reconstruction`, `stable-diffusion`, `unet`, `pytorch`
3. Consider adding a LICENSE file (MIT, Apache 2.0, etc.)

---

**Need more details?** See `GITHUB_PUSH_GUIDE.md` for comprehensive instructions.

