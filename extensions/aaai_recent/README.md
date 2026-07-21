# Recent AAAI baseline extension

This directory is reserved for the two-method extension described in
[`docs/AAAI_RECENT_BASELINES.md`](../../docs/AAAI_RECENT_BASELINES.md).

The extension is intentionally separate from the active no-KNN 15-method
`protocol.json`. Until `protocol_extension.json` reaches `frozen` status, its
models may be used only for implementation tests and official-Tuning studies,
not for Eval reporting.

Implemented extension identifiers:

- `DPAD_AAAI24`
- `DNE_AAAI25`

The paper-derived model definitions are in `models.py`; the single-unit adapter
is `run_one_recent.py`. The runner defaults to `--profile formal` and refuses
to execute while `protocol_extension.json` is not frozen. This prevents an
implementation smoke configuration from being used accidentally on Eval.

Local interface validation uses the explicit `--profile smoke` path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_recent_smoke.ps1
& 'C:\Users\wyyxx\.conda\envs\00gwk\python.exe' `
  scripts\export_recent_smoke_evidence.py
```

The smoke profile reduces both methods to three epochs and is never eligible
for a formal table. Its 4/4 runtime-free evidence is stored in
`validation/aaai_recent_local_00gwk_smoke.json`. Formal completion still
requires official-Tuning selection, frozen parameter values and hashes, and a
separate controller/results root for the 386 extension units.
