# Zenodo Publication Guide

Guide for archiving your code on Zenodo and obtaining a DOI for academic citations.

## Why Use Zenodo?

- **Permanent Archive**: Your code is preserved long-term, even if GitHub goes down
- **DOI (Digital Object Identifier)**: Makes your code citable in academic papers
- **Version Control**: Each release gets its own DOI
- **Free and Open**: Maintained by CERN, no cost
- **Academic Recognition**: Zenodo archives count as research outputs

## Prerequisites

- [x] GitHub repository published (✓ https://github.com/BNUL/PATH_LiDAR)
- [ ] GitHub account
- [ ] ORCID account (optional but recommended)

## Step-by-Step Instructions

### Step 1: Create Zenodo Account

1. Go to https://zenodo.org/
2. Click "Sign up" or "Log in"
3. Choose "Log in with GitHub" (easiest option)
4. Authorize Zenodo to access your GitHub account

### Step 2: Link GitHub Repository

1. After logging in, go to https://zenodo.org/account/settings/github/
2. You'll see a list of your GitHub repositories
3. Find "BNUL/PATH_LiDAR" in the list
4. Click the **ON** toggle switch to enable archiving
5. The repository is now linked!

### Step 3: Create a GitHub Release

Zenodo archives are triggered by GitHub releases. Create your first release:

1. Go to https://github.com/BNUL/PATH_LiDAR
2. Click "Releases" (right sidebar) → "Create a new release"
3. Fill in the release information:
   - **Tag version**: `v1.0.0`
   - **Release title**: `v1.0.0 - Initial Release`
   - **Description**: 
     ```markdown
     ## Initial Release of PATH LiDAR Simulator
     
     Full-waveform LiDAR simulator for vegetation canopy analysis.
     
     ### Features
     - Multiple crown shapes (cylinder, sphere, cone)
     - Gap probability modeling
     - Multiple scattering effects
     - RAMI validation dataset
     - Sensitivity analysis tools
     
     ### Files Included
     - 4 Python modules
     - 9 RAMI validation datasets
     - Complete documentation
     - Example scripts
     ```
4. Click "Publish release"

### Step 4: Verify Zenodo Archive

1. Wait a few minutes for Zenodo to process
2. Go to https://zenodo.org/account/settings/github/
3. You should see your release listed
4. Click on it to view the Zenodo record
5. **Copy the DOI badge code** (you'll add this to README)

### Step 5: Update README with DOI Badge

After getting your DOI, add this near the top of README.md:

```markdown
# LiDAR Waveform Simulator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

Replace `XXXXXXX` with your actual DOI number.

### Step 6: Update ORCID (Optional but Recommended)

If you have an ORCID:

1. Get your ORCID at https://orcid.org/ (free registration)
2. Update `.zenodo.json` file with your real ORCID
3. Update `CITATION.cff` file with your real ORCID
4. Commit and push changes
5. Create a new release (v1.0.1) to update Zenodo

## .zenodo.json File

I've created a `.zenodo.json` file that provides metadata to Zenodo. Update the ORCID field:

```json
"orcid": "0000-0001-2345-6789"  // Replace with your real ORCID
```

## How to Cite After Zenodo Publication

After getting your DOI, update the Citation section in README.md:

```bibtex
@software{li2026path,
  title={PATH LiDAR: Full-Waveform LiDAR Simulator for Vegetation Canopy Analysis},
  author={Li, Weihua},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXXX},
  url={https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## Updating Your Software

When you make updates:

1. Make changes to your code
2. Commit and push to GitHub
3. Create a new release (e.g., v1.1.0)
4. Zenodo automatically archives the new version
5. Each version gets its own DOI
6. The main DOI always points to the latest version

## Benefits for Academic Career

✅ **Citable Research Output**: DOI makes it citeable in papers  
✅ **Impact Metrics**: Track downloads and citations  
✅ **CV Material**: List as a published research output  
✅ **Grant Applications**: Show research dissemination  
✅ **Reproducibility**: Permanent archive ensures long-term access  

## Zenodo vs GitHub

| Feature | GitHub | Zenodo |
|---------|--------|--------|
| Version Control | ✓ | ✗ |
| Collaboration | ✓ | ✗ |
| DOI | ✗ | ✓ |
| Permanent Archive | ✗ | ✓ |
| Academic Citations | Informal | Formal |
| Free | ✓ | ✓ |

**Best Practice**: Use both! GitHub for development, Zenodo for archiving.

## Next Steps

1. [ ] Create Zenodo account
2. [ ] Link GitHub repository
3. [ ] Create first GitHub release (v1.0.0)
4. [ ] Verify Zenodo archive
5. [ ] Get DOI
6. [ ] Update README with DOI badge
7. [ ] Update ORCID in `.zenodo.json` (if you have one)
8. [ ] Share your DOI in papers and presentations

## Questions?

- Zenodo Help: https://help.zenodo.org/
- GitHub Releases: https://docs.github.com/en/repositories/releasing-projects-on-github

---

**Current Status**: `.zenodo.json` file created and ready for Zenodo integration!

**Your Repository**: https://github.com/BNUL/PATH_LiDAR
