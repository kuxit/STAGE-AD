# Validation evidence

This directory contains small, reviewable evidence generated from local smoke
tests. It is safe to version because it contains metrics and environment
metadata only: no raw dataset, score array, checkpoint, password, or SSH key.

`local_00gwk_smoke.json` proves that all 15 active methods completed on one
official U-series and one official M-series with six finite metrics under seed
2026. It does **not** establish paper-ready accuracy or efficiency. Runtime is
intentionally omitted because the laptop was not an exclusive benchmark host.

Regenerate it after a successful smoke test:

```powershell
& 'C:\Users\wyyxx\.conda\envs\00gwk\python.exe' `
  scripts\export_smoke_evidence.py
```

`aaai_recent_local_00gwk_smoke.json` is the corresponding runtime-free record
for `DPAD_AAAI24` and `DNE_AAAI25`: 2 methods x 2 tracks = 4 valid records.
It proves only that the paper-derived implementations train, score a complete
series, and emit the official six metrics. It is not eligible for formal Eval
because the official-Tuning configuration has not yet been frozen.

The historical smoke directory may still contain two KNN records. The evidence
exporter follows the active controller and intentionally omits them.

Regenerate it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_recent_smoke.ps1
& 'C:\Users\wyyxx\.conda\envs\00gwk\python.exe' `
  scripts\export_recent_smoke_evidence.py
```
