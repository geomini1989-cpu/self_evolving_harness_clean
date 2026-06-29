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

Real-LLM multi-round self-evolution:

```powershell
python main_loop.py --mode llm-evolve --sample-size 60 --epochs 3
```

This mode uses the configured API for batch extraction, bad-case attribution, and patch generation. It keeps the token-saving router/cache/tip-memory safeguards, but it does not use the local benchmark fast path.
For safety, this mode uses stricter memory promotion than the cheap demo path: a new tip usually needs repeated evidence, or confidence at least 0.95, before it writes long-term Prompt/Skill/Few-shot memory.
Non-schema bad cases use a few-shot-first policy: the harness tries to add a low-risk example to `memory/examples.json` before changing global Prompt or Skill memory.

Fresh 100-sample evolution demo from empty runtime memory:

```powershell
python main_loop.py --mode llm-evolve --sample-size 100 --epochs 3 --batch-size 8 --max-workers 2 --reset-state
```

To force one bad case while still using the real LLM for reflection and patch generation:

```powershell
python main_loop.py --mode evolve-demo --use-llm-evolution
```

Do not commit `.env` or generated runtime/cache files.

## Transfer test runner

Run this to verify that the harness abstraction can transfer to similar tasks without changing the core loop:

```powershell
python tools/transfer_test_runner.py
```

The runner covers text complaints, image metadata QA, audio transcript routing, and recommendation event streams through the same `state -> action -> feedback` interface. It writes:

- `memory/transfer_report.json`
- `memory/transfer_traces.jsonl`

You can also run one suite only:

```powershell
python tools/transfer_test_runner.py --suite visual_quality
```

## Dashboard

```powershell
streamlit run dashboard/app.py
```
