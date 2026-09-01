# Setup Guide: Cross-OEM fine-tuning

## Why? 

This is the simplest test I could think of for integrating fine-tuning with the current GPUs on our demo environment. One aspect I'd like to add is a 2-node A100 test with a data-parallel section. The idea is that if we can resume from checkpoints across different OEMs, we've effectively written a unified compute layer for our current workload.
There's a custom layer with a transfer schema that's agnostic to the machines available. It's worth noting that this setup demonstrates cross-OEM compute that is inherently sequential — i.e., we cannot split batches of training across machines. One way to work around this is to define many jobs, each of which can run on a single machine. However, that's likely the next step; for now, this repo exists purely for informative purposes, to guide our next steps.

I purposefully kept the flow limited to resumption from checkpoints, since that's the simplest thing to demonstrate and control at this stage, IMO. We could later add disaggregated prefill for inference, plus a data-parallel stage across the two A100s in the demo env, just to show both are feasible.

I also used Dagster to see if it's a useful tool for us to integrate into our workflow. The web UI is quite nice, and it can also integrate with Dagster Cloud for team-wide use. For now, just use localhost to test.


## Prerequsites 

```bash
# Homebrew (macOS) or use your OS package manager
brew install python@3.12          # macOS
sudo apt-get install python3.12   # Ubuntu/Debian

# Conda (Miniconda or Anaconda)
brew install --cask miniconda     # macOS

# Git (optional, for cloning)
brew install git                  # macOS
```

---

## Setup

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

Click launch runs to run the workflow:

<img width="1713" height="935" alt="Screenshot 2026-09-01 at 20 41 11" src="https://github.com/user-attachments/assets/69d5e1fd-6519-4900-b193-25d0b216b93d" />



