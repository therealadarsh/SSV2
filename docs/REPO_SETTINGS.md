# Repository settings

In a public repo, other people can **read and fork** the code, open issues and open pull
requests. They **cannot** push, merge, delete branches or change settings unless you add
them as collaborators. So the main risks are your own mistakes, a coding agent pushing
somewhere it shouldn't, leaked secrets, and malicious pull requests from forks. These
settings cover all four. Total time: about 10 minutes.

## 1. Import the rulesets (Settings → Rules → Rulesets)

Choose **New ruleset → Import a ruleset** and upload each file:

| File | What it enforces |
| --- | --- |
| `.github/rulesets/protect-main.json` | `main` can't be deleted or force-pushed; every change needs a PR; squash merge only; review threads must be resolved |
| `.github/rulesets/protect-release-tags.json` | Release tags `v*` can't be deleted or moved once pushed |

Neither ruleset has bypass actors, so the rules apply to you too. That's deliberate.

**After Phase 0's CI is merged**, edit "Protect main", add **Require status checks to pass**,
and select `lint`, `typecheck`, `test`, `docker` and `eval`. The `eval` job exists as of Phase 2.
Don't add a check that never runs: a required check with no workflow blocks every merge.

## 2. General (Settings → General)

- Pull requests: allow **squash merging** only; turn on **Automatically delete head branches**
- Features: turn off **Wikis** and **Projects** unless you use them; keep **Issues** on
- Leave **Allow forking** on (it's normal for public repos and can't harm your copy)

## 3. Actions (Settings → Actions → General)

- **Fork pull request workflows from outside collaborators:** "Require approval for all
  outside collaborators". Strangers' PRs can't run code in your CI until you approve them.
- **Workflow permissions:** "Read repository contents and packages permissions". Each
  workflow asks for more in its own `permissions:` block.
- Leave **Allow GitHub Actions to create and approve pull requests** off.

## 4. Code security (Settings → Code security)

Turn on all of these (free for public repos):

- Private vulnerability reporting (used by SECURITY.md)
- Dependabot alerts and Dependabot security updates
- Secret scanning and **push protection**, which blocks pushes that contain API keys

## 5. Collaborators (Settings → Collaborators)

Add nobody for now. If you do later, give **Write** access, never **Admin**. Admins can
change rulesets and delete the repo.

## 6. Account

Turn on two-factor authentication for your GitHub account. Anyone who gets into your
account can delete the repo regardless of these settings.
