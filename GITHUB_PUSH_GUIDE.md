# 🚀 Guide: Push Forensic Reconstruction to GitHub

## Step-by-Step Instructions

### **Step 1: Create New Repository on GitHub**

1. Go to https://github.com/new (or click "New repository" on GitHub)
2. **Repository name**: `forensic-reconstruction` (or `forensic_reconstruction`)
3. **Description**: "AI-powered forensic facial reconstruction system using Stable Diffusion and U-Net"
4. **Visibility**: 
   - Choose **Public** (if you want others to see it)
   - Or **Private** (if you want it private)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **"Create repository"**

### **Step 2: Get Your Repository URL**

After creating, GitHub will show you a URL like:
```
https://github.com/85-Astatine/forensic-reconstruction.git
```

**Copy this URL** - you'll need it in Step 4!

### **Step 3: Stage and Commit Your Files**

Run these commands in your terminal:

```bash
cd /home/teaching/G14/forensic_reconstruction

# Stage all changes
git add .

# Commit with a message
git commit -m "Add forensic reconstruction system with Stable Diffusion and U-Net"
```

### **Step 4: Update Remote and Push**

```bash
# Remove old remote (if exists)
git remote remove origin

# Add your new repository as remote
git remote add origin https://github.com/85-Astatine/forensic-reconstruction.git

# Push to GitHub
git push -u origin main
```

### **Step 5: Authenticate**

When you run `git push`, GitHub will ask for authentication. You have two options:

#### **Option A: Personal Access Token (Recommended)**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "Forensic Reconstruction Push"
4. Select scopes: Check `repo` (full control)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. When `git push` asks for password, paste the token instead

#### **Option B: SSH Key (More Secure)**
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to GitHub: https://github.com/settings/keys
3. Use SSH URL: `git@github.com:85-Astatine/forensic-reconstruction.git`

---

## What Files Will Be Pushed?

✅ **Will be pushed:**
- All source code (`src/`)
- Documentation (`README.md`, `SYSTEM_ANALYSIS.md`, etc.)
- Configuration files (`requirements.txt`, etc.)
- Scripts (`scripts/`)

❌ **Will NOT be pushed** (excluded by .gitignore):
- `venv/` (virtual environment)
- `models/` (large model files)
- `output/` (generated results)
- `logs/` (log files)
- `__pycache__/` (Python cache)
- Large image files

---

## Troubleshooting

### "Repository not found"
- Make sure you created the repo on GitHub first
- Check the repository name matches exactly
- Verify you're logged into the correct GitHub account

### "Authentication failed"
- Use Personal Access Token instead of password
- Make sure token has `repo` scope

### "Large file detected"
- GitHub has a 100MB file size limit
- Large model files are already excluded in .gitignore
- If you see this error, check what file is too large: `git ls-files | xargs ls -lh | sort -k5 -hr | head -10`

### "Branch 'main' does not exist"
- Your local branch might be named `master`
- Check: `git branch`
- If it's `master`, rename: `git branch -M main`

---

## After Pushing

1. Visit your repository: `https://github.com/85-Astatine/forensic-reconstruction`
2. Add a README description
3. Add topics/tags: `forensic`, `face-reconstruction`, `stable-diffusion`, `unet`, `pytorch`
4. Consider adding a license (MIT, Apache 2.0, etc.)

---

## Quick Command Summary

```bash
# 1. Stage files
git add .

# 2. Commit
git commit -m "Initial commit: Forensic reconstruction system"

# 3. Update remote (replace with YOUR repo URL)
git remote set-url origin https://github.com/85-Astatine/forensic-reconstruction.git

# 4. Push
git push -u origin main
```

---

**Need help?** Check the error message and refer to the troubleshooting section above.

