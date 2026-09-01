# Setup Guide: cross-vendor-mooncake-test

## 🛠️ Prerequisites (install once)

```bash
# Homebrew (macOS) or use your OS package manager
brew install python@3.12          # macOS
sudo apt-get install python3.12   # Ubuntu/Debian

# Conda (Miniconda or Anaconda)
brew install --cask miniconda     # macOS

# Git (optional, for cloning)
brew install git                  # macOS
```

**Connected integrations:**
- MCP: `perplexity_search` — Connected
- LSP: disabled

---

## 🚀 Setup (≤ 50 lines)

```bash
# 1. Ensure Python 3.12 is the default python3
python3 --version || { echo "Install Python 3.12 first"; exit 1; }

# 2. Create a dedicated conda env (named after the project)
conda create -y -n cross-vendor-mooncake-test python=3.12
conda activate cross-vendor-mooncake-test

# 3. Install Dagster + webserver (add any extra deps required by the pipeline)
pip install --upgrade pip setuptools wheel
pip install dagster dagster-webserver

# 4. (Optional) Persist Dagster state across runs
export DAGSTER_HOME="${HOME}/.dagster_home"
mkdir -p "$DAGSTER_HOME"

# 5. Clone the repo (skip if already present)
git clone https://github.com/<your-org>/cross-vendor-mooncake-test.git
cd cross-vendor-mooncake-test

# 6. Run Dagster — the CLI entry point is provided by the installed package
dagster dev -f dagster_basic_portability.py
```

**Expected log fragment:**
```
Serving dagster-webserver on http://127.0.0.1:3000 ...
```

## 7. Open the UI

In a browser, go to: [http://127.0.0.1:3000](http://127.0.0.1:3000)


