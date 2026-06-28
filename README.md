# self_evolving_harness

Clean, token-optimized version of the self-evolving customer complaint harness.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python main_loop.py
```

Do not commit `.env` or generated runtime/cache files.
