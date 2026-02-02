# GitHub Publication Guide

Step-by-step instructions for publishing this project to GitHub.

## Before You Start

### 1. Update Personal Information

Replace placeholder information in these files:

**LICENSE**
- Line 3: Replace `[Your Name]` with your actual name

**CITATION.cff**
- Lines 7-10: Update author information
  - `family-names`: Your family/last name
  - `given-names`: Your first/given name
  - `affiliation`: Your institution
  - `orcid`: Your ORCID identifier (get one at https://orcid.org/)
- Line 11: Update repository URL
- Line 12: Update project URL

**README.md**
- Line 159: Update GitHub clone URL with your username
- Line 190: Update citation with your name and year
- Line 199: Update contact email

### 2. Review Code Comments

The Python files contain some Chinese comments. Consider translating them to English for international audience, or keep them as-is for bilingual documentation.

## Publication Steps

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Choose a repository name (e.g., `lidar-waveform-simulator`)
3. Add a description: "Full-waveform LiDAR simulator for vegetation canopy analysis"
4. Choose visibility: **Public**
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Step 2: Initialize Local Git Repository

Open PowerShell in your project folder and run:

```powershell
cd "d:\PhD\matlab\RSE_LiDAR\PATH_LiDAR\english_version_backup"
git init
```

### Step 3: Configure Git (First Time Only)

```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Step 4: Add Files to Git

```powershell
# Add all project files
git add .

# Or add files selectively:
git add README.md
git add LICENSE
git add .gitignore
git add requirements.txt
git add CITATION.cff
git add CONTRIBUTING.md
git add RAMI_DATA_README.md
git add *.py
git add RAMI_*
```

### Step 5: Create Initial Commit

```powershell
git commit -m "Initial commit: LiDAR waveform simulator with RAMI validation data"
```

### Step 6: Connect to GitHub

Replace `yourusername` with your GitHub username:

```powershell
git remote add origin https://github.com/yourusername/lidar-waveform-simulator.git
git branch -M main
```

### Step 7: Push to GitHub

```powershell
git push -u origin main
```

You may be prompted for GitHub credentials. Use a Personal Access Token (PAT) instead of password.

### Step 8: Verify Upload

1. Visit your repository: `https://github.com/yourusername/lidar-waveform-simulator`
2. Check that all files are present
3. Verify README.md displays correctly
4. Check that RAMI_* folders are uploaded

## Post-Publication Tasks

### Add Topics/Tags

On your GitHub repository page:
1. Click "⚙️ Settings" (or the gear icon near "About")
2. Add topics: `lidar`, `remote-sensing`, `python`, `vegetation`, `canopy-structure`, `waveform-simulation`

### Create a Release

1. Go to "Releases" → "Create a new release"
2. Tag version: `v1.0.0`
3. Release title: "v1.0.0 - Initial Release"
4. Description: Summarize features and capabilities
5. Click "Publish release"

### Enable GitHub Pages (Optional)

If you want to host documentation:
1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: main / docs folder
4. Save

### Add Repository Description

On the main page:
1. Click "⚙️" next to "About"
2. Description: "Full-waveform LiDAR simulator for vegetation canopy analysis using PATH model"
3. Website: Your project page (if any)
4. Topics: Add relevant keywords

## File Size Considerations

GitHub has limits:
- Individual files: 100 MB max
- Repository: 1 GB recommended

If your RAMI data folders are large:
- Consider using Git LFS (Large File Storage)
- Or provide download links in README

### Using Git LFS (if needed)

```powershell
git lfs install
git lfs track "*.txt"
git lfs track "*.gr#"
git lfs track "*.grf"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

## Updating the Repository

After making changes:

```powershell
git add .
git commit -m "Description of changes"
git push
```

## Common Issues

### Issue: Large files rejected
**Solution**: Use Git LFS or provide external download links

### Issue: Authentication failed
**Solution**: Use Personal Access Token (PAT) instead of password
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

### Issue: Files in wrong structure
**Solution**: 
```powershell
git rm -r --cached .
git add .
git commit -m "Fix file structure"
git push
```

## Recommended GitHub Settings

- Enable Issues (for bug reports)
- Enable Discussions (for Q&A)
- Add branch protection rules (for collaborative projects)
- Add a Code of Conduct
- Add security policy

## Promoting Your Repository

1. Share on social media (Twitter, LinkedIn)
2. Submit to relevant lists (Awesome lists)
3. Add to your institutional page
4. Present at conferences
5. Write a blog post
6. Submit to Papers with Code (if applicable)

## Questions?

- GitHub Help: https://docs.github.com/
- Git Tutorial: https://git-scm.com/docs/gittutorial
- Git LFS: https://git-lfs.github.com/

---

**Ready to publish?** Make sure you've:
- ✅ Updated all personal information
- ✅ Tested code runs correctly
- ✅ Reviewed all documentation
- ✅ Chosen appropriate license
- ✅ Created GitHub account

Good luck with your open source project! 🚀
