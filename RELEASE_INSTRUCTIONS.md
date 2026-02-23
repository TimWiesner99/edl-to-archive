# Release Instructions

This guide walks you through creating releases for the EDL-to-Archive application. The GitHub Actions workflow automatically packages the source code with launcher scripts and attaches a zip to the release.

## Prerequisites

- Your GitHub repository must be **public** (so the auto-update checker works without authentication).
- GitHub Actions is enabled by default on public repositories.

## How it works

1. You tag a commit with a version number (e.g., `v1.0.0`).
2. Pushing the tag triggers the GitHub Actions workflow (`.github/workflows/build.yml`).
3. The workflow packages all source files with the launcher scripts (`run.bat`, `run.sh`) into a zip.
4. The zip is attached to a GitHub Release.
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
4. The packaging step only takes a few seconds.

### 4. Check the release

1. Go to the **Releases** section of your repository.
2. You should see a new release with:
   - Auto-generated release notes (list of commits since last tag).
   - One zip file: `EDL-to-Archive.zip`.

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

Send your colleagues the `EDL-to-Archive.zip` file. They need to:

1. Extract the zip to any folder.
2. **Windows**: Double-click `run.bat`.
3. **macOS / Linux**: Double-click `run.sh` (or run `./run.sh` in a terminal).

The launcher script automatically installs [uv](https://docs.astral.sh/uv/) (if needed), downloads the correct Python version, installs dependencies, and starts the app. The first launch takes a moment; subsequent launches are near-instant.

## Troubleshooting

### The Actions workflow doesn't run

- Make sure you pushed the **tag**, not just the commit: `git push origin v1.0.0`
- Check that the tag name starts with `v` (the workflow triggers on `v*`)

### Users don't see update notifications

- The repository must be public for unauthenticated API access.
- The `GITHUB_REPO` constant in `src/updater.py` must match your actual repository path (e.g., `"TimWiesner99/edl-to-archive"`).
- The tag must follow the `vX.Y.Z` format and be a proper release (not a draft).
