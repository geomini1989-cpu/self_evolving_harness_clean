# self_evolving_harness

Clean, token-optimized version of the self-evolving customer complaint harness.

## Competition direction

This project targets **题目1：算法方向** from the technical competition material.

Paper-inspired direction:

- **SkillOS**: curate and retrieve task-specific skills from `memory/SKILL.md`.
- **SkillOpt**: add an execution strategy layer that gates LLM outputs with skill-router confidence.
- **Self-Harness**: keep the `execute -> evaluate -> reflect -> evolve` loop.

The base LLM is frozen. The system improves F1 through framework-level optimization:

- local skill routing instead of extra scout LLM calls
- persistent deterministic LLM cache
- schema repair and confidence-gated F1 post-processing
- bad-case attribution, skill evolution, sample-first regression, and rollback
- Streamlit dashboard for F1 / evolution trace

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python main_loop.py
```

If `.env` has an empty `API_KEY`, the harness runs in offline demo mode with a local rule-based model. Add a valid API key to switch to the real LLM.

## Run modes

Fast closed-loop demo, suitable for live presentation:

```powershell
python main_loop.py --mode demo
```

Full-dataset benchmark, suitable for final F1 reporting:

```powershell
python main_loop.py --mode benchmark
```

Benchmark mode is token-conscious by default: it evaluates the full dataset once, uses larger batches, keeps cache enabled, disables few-shot injection, and skips the attribution/evolution step.

If you want a cheaper benchmark rehearsal before the full run:

```powershell
python main_loop.py --mode benchmark --sample-size 1000
```

Do not commit `.env` or generated runtime/cache files.

## Dashboard

```powershell
streamlit run dashboard/app.py
```
