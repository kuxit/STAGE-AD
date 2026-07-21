# Local compatibility adapters

Files in this directory are used only by portability smoke tests. They are
never authorized for the formal seed=2027 experiment.

`paano_official_one_windows.py` is byte-for-byte equal to the frozen PaAno
adapter except that it passes a POSIX-formatted path to the released PaAno
loader. The upstream loader uses `file_path.split('/')`, which cannot parse a
native Windows path. Model code, data boundary, seed, hyperparameters, scoring,
and six-metric evaluation are unchanged.
