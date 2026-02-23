# Release Instructions

This guide walks you through creating releases for the EDL-to-Archive desktop application. The GitHub Actions workflow automatically builds executables for Windows and macOS and attaches them to the release.

## Prerequisites

- Your GitHub repository must be **public** (so the auto-update checker works without authentication).
- GitHub Actions is enabled by default on public repositories.

## How it works

1. You tag a commit with a version number (e.g., `v1.0.0`).
2. Pushing the tag triggers the GitHub Actions workflow (`.github/workflows/build.yml`).
3. The workflow builds the app on Windows and macOS using PyInstaller.
4. The built executables are zipped and attached to a GitHub Release.
5. When users open the app, it checks for the latest release and notifies them if an update is available.

## Step-by-step: Creating a release

### 1. Update the version number

Edit `pyproject.toml` and update the `version` field:

```toml
[project]
version = "1.0.0"
```

Also update `src/updater.py` — the `get_current_version()` fallback string should match:

```python
return "1.0.0"
```

Commit these changes:

```bash
git add pyproject.toml src/updater.py
git commit -m "Bump version to 1.0.0"
```

### 2. Create and push a tag

```bash
git tag v1.0.0
git push origin main
git push origin v1.0.0
```

### 3. Watch the build

1. Go to your repository on GitHub.
2. Click the **Actions** tab.
3. You should see a workflow run triggered by the tag push.
4. The build takes a few minutes (it builds on both Windows and macOS).

### 4. Check the release

1. Go to the **Releases** section of your repository (right sidebar on the main page, or the "Releases" link).
2. You should see a new release with:
   - Auto-generated release notes (list of commits since last tag).
   - Two zip files: `EDL-to-Archive-Windows.zip` and `EDL-to-Archive-macOS.zip`.

### 5. (Optional) Edit the release notes

Click "Edit" on the release to add a human-readable summary of changes. This text is shown to users in the update notification dialog.

## How the auto-update works

When the application starts, it makes a request to:

```
https://api.github.com/repos/TimWiesner99/edl-to-archive/releases/latest
```

It compares the `tag_name` from the response against the app's built-in version. If the release is newer, a dialog is shown with:

- The current vs. new version numbers
- The release notes (what you write on the GitHub Release page)
- A "Download" button that opens the release page in the browser
- A "Skip This Version" button to suppress the notification for that version
- A "Remind Me Later" button to dismiss for the current session

The check has a 3-second timeout and fails silently if the user is offline.

## Distributing to users

Send your colleagues the appropriate zip file:

- **Windows users**: `EDL-to-Archive-Windows.zip` — extract and run `EDL-to-Archive.exe`
- **Mac users**: `EDL-to-Archive-macOS.zip` — extract and run the `EDL-to-Archive` app

On macOS, users may need to right-click → Open the first time (Gatekeeper warning for unsigned apps). If you want to avoid this, you can sign the app with an Apple Developer certificate — but that's a separate process and not required for internal distribution.

## Troubleshooting

### The Actions workflow doesn't run

- Make sure you pushed the **tag**, not just the commit: `git push origin v1.0.0`
- Check that the tag name starts with `v` (the workflow triggers on `v*`)

### The build fails

- Click into the failed workflow run in the Actions tab to see logs.
- Common issues: Python version not available, dependency installation failures.
- The workflow uses Python 3.14 — if that's not available on GitHub Actions yet, change `python-version` in the workflow to a version that is (e.g., `"3.12"`).

### Users don't see update notifications

- The repository must be public for unauthenticated API access.
- The `GITHUB_REPO` constant in `src/updater.py` must match your actual repository path (e.g., `"TimWiesner99/edl-to-archive"`).
- The tag must follow the `vX.Y.Z` format and be a proper release (not a draft).
