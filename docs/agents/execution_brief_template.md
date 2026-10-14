# Execution Brief — CPS-Paper2 V20

> **Author:** Academic Lead  
> **Phase / slice:** e.g. `Phase 2.1 — aci_perceptual_shield.py`  
> **Date:** YYYY-MM-DD  
> **Status:** Draft | Ready for execution

## 1. Objective (one sentence)

…

## 2. Scientific linkage

- Bottleneck addressed: (cold-start | coverage drift | per-chunk VMAF / masking)
- Roadmap ref: `docs/help.md` § …

## 3. Scope

### In scope

- …

### Out of scope (explicit)

- …

## 4. Files to touch

| Action | Path | Notes |
|--------|------|--------|
| create / edit / read-only | `…` | … |

**Frozen:** `paper_v1/**` — no writes.

## 5. Implementation requirements

### Interfaces / behavior

- …

### Defaults (must match spec unless noted)

- …

### Reference templates (read-only)

- `@src/training/certified_perceptual_shield.py`
- …

## 6. Invariants checklist

- [ ] Episode seeds `1000 + ep` preserved for any eval added
- [ ] No changes to `abr_multi_env_v18.py` unless listed above
- [ ] Diagnostics in `info` dict as per roadmap

## 7. Acceptance criteria

| ID | Criterion | How to verify |
|----|-----------|----------------|
| AC-1 | … | command / assertion |

## 8. Verification command(s)

```bash
# copy exact commands from roadmap
```

## 9. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| … | … |

## 10. Handoff message (copy to Execution Arm chat)

```text
[Paste this block for the executor agent.]
```
