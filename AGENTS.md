# Agent rules (sticky tech-demos)

Short rules for future cloud agents working in this monorepo.

1. **Only touch `apps/`** (and root docs when the task explicitly needs them). Do not invent new top-level product trees.
2. **Never invent secrets.** Do not commit `.env` files with keys, tokens, or HF credentials. Document required env vars in each app README; use mock/offline paths when credentials are missing.
3. **Every PR needs screenshot + video** of the running app(s) you changed. Put artifacts under that app's `artifacts/` (or the run's artifact store) and link them from the PR description.
4. Prefer Bun for JS/TS demos. Prefer honest dry-run / mock paths over claiming GPU demos work without weights.
5. Keep this ONE monorepo forever — add demos under `apps/<slug>/`, do not spawn one repo per demo.
