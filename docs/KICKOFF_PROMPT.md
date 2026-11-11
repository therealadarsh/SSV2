# Kickoff prompt

Start a new Claude Code session on `https://github.com/therealadarsh/SSV2` in the **Dev**
cloud environment (it allows Hugging Face and PyTorch downloads), then paste:

---

I'm starting SSV2, a rewrite of my Semantic-Search project
(https://github.com/therealadarsh/Semantic-Search, branch `claude/jolly-planck-wlqzqt`),
as a retrieval service that measures its own quality. I'm doing this to learn modern ML
engineering, CI/CD and system design, so explain your choices as you go.

1. Read `CLAUDE.md`, `docs/PLAN.md` and `docs/DECISIONS.md` in this repo. They hold the plan
   and the decisions made so far.
2. Do **Phase 0** from the plan, on a branch, never on `main`:
   - `src/ssv2` package, `pyproject.toml` and `uv.lock` (CPU-only PyTorch from the PyTorch index)
   - FastAPI app with `/health` and `/ready`; ruff, a type checker, pytest, pre-commit and a Makefile
   - Multi-stage Dockerfile, CI workflow, release workflow and Dependabot. Adapt them from
     Semantic-Search's `Dockerfile` and `.github/` on that branch and read its
     `docs/CICD_GUIDE.md`, but apply the improvements listed in PLAN.md (actions pinned to
     commit SHAs, type checking, Trivy scan).
3. Run everything locally before pushing (lint, types, tests, Docker build, smoke test), and
   show me the results.
4. Push the branch and open a PR. Then tell me which status checks to add to the
   "Protect main" ruleset.

Stop after Phase 0's gate so I can review before Phase 1.
