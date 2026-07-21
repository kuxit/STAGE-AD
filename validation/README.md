# Validation evidence

This directory contains small, reviewable evidence generated from local smoke
tests. It is safe to version because it contains metrics and environment
metadata only—no raw dataset, score array, checkpoint, password, or SSH key.

`local_00gwk_smoke.json` proves that all 16 frozen methods completed on one
official U-series and one official M-series with six finite metrics under seed
2027. It does **not** establish paper-ready accuracy or efficiency. Runtime is
intentionally omitted because the laptop was not an exclusive benchmark host.

Regenerate it after a successful smoke test:

```powershell
& 'C:\Users\wyyxx\.conda\envs\00gwk\python.exe' `
  scripts\export_smoke_evidence.py
```
