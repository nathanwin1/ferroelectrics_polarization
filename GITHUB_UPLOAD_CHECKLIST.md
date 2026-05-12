# GitHub Upload Checklist

## 1) Verify repo contents

- Keep source code: `test1_modular/`, `run_*.py`, `bezier_classify.py`, `postprocess_bezier.py`
- Keep docs/config: `README.md`, `.gitignore`, `requirements.txt`, `LICENSE` (if added)
- Do not upload caches/outputs: `__pycache__/`, `.venv/`, generated `.png/.mp4/.npz`

## 2) Initialize git (if not already initialized)

```bash
git init
```

## 3) Confirm what will be committed

```bash
git status
```

## 4) Stage and commit

```bash
git add .
git commit -m "Initial commit: ferroelectrics simulation and postprocessing workflow"
```

## 5) Create GitHub repo and connect remote

Replace `<your-repo-url>` with your repository URL:

```bash
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## 6) Post-push quick checks

- Open `README.md` on GitHub and confirm formatting
- Confirm only intended files are present
- Confirm no generated outputs/caches were uploaded
