# Recent AAAI baseline extension

This directory is reserved for the two-method extension described in
[`docs/AAAI_RECENT_BASELINES.md`](../../docs/AAAI_RECENT_BASELINES.md).

The extension is intentionally separate from the completed 14-method no-KNN
baseline root. Its formal contract is frozen in
[`configs/aaai_recent_locked_seed2026.json`](../../configs/aaai_recent_locked_seed2026.json).
The controller first qualifies both fixed configurations on official TSB-AD
Tuning, then evaluates both methods on all 193 Eval series. Only after all 386
new Eval units and all 2,702 original units pass strict validation can STAGE
Eval begin.

Implemented extension identifiers:

- `DPAD_AAAI24`
- `DNE_AAAI25`

The paper-derived model definitions are in `models.py`; the single-unit adapter
is `run_one_recent.py`; and `scripts/recent_baseline_autopilot.py` provides the
atomic, resumable two-phase controller. The runner defaults to `--profile
formal` and validates the lock, source hashes, official allowlist, split,
physical GPU binding, and output identity before accepting a unit.

Local interface validation uses the explicit `--profile smoke` path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_recent_smoke.ps1
& 'C:\Users\wyyxx\.conda\envs\00gwk\python.exe' `
  scripts\export_recent_smoke_evidence.py
```

The smoke profile reduces both methods to three epochs and is never eligible
for a formal table. Its 4/4 runtime-free evidence is stored in
`validation/aaai_recent_local_00gwk_smoke.json`. Formal Tuning contains 44
fixed-configuration qualification units (22 series x 2 methods); it performs
no ranking or parameter selection. Eval contains 386 units and cannot feed
back into tuning. Shared-run timing remains diagnostic and is ineligible for
the paper efficiency table.

All 16 baselines are run and audited. If the final paper reports 15 baselines,
the omitted method must be chosen by a predeclared relevance or implementation-
provenance criterion, never by its observed accuracy.
