# GitHub Upload Guide

## One-time setup
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Create & push
1. github.com → **New repository** → name: `AI-Cybersecurity-Threat-Detection`
   - Description: *ML-based network intrusion & anomaly detection with simulated attacks (DoS, brute force, port scan, exfiltration), severity-ranked alerts and a Flask API.*
   - Public, add MIT license, do NOT add a README (you already have one).
2. In your project folder:
```bash
git init
git add .
git commit -m "chore: initial project structure"
git branch -M main
git remote add origin https://github.com/<your-username>/AI-Cybersecurity-Threat-Detection.git
git push -u origin main
```
3. Topics (repo → ⚙ next to About): `cybersecurity` `machine-learning` `intrusion-detection` `anomaly-detection` `python` `scikit-learn` `random-forest` `isolation-forest` `threat-detection` `flask` `data-science` `soc`
4. Commit images: run `python main.py run`, then `git add images/ && git commit -m "docs: add result charts"`. Large raw datasets stay out (`.gitignore`).
5. Pin the repo on your profile, add a 60–90 s screen recording (upload to YouTube unlisted or drag a GIF into the README).

## Commit message style
`feat:` new functionality · `fix:` bug · `docs:` README/images · `refactor:` · `test:` · `chore:`
