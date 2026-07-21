# Recent AAAI baseline extension

This directory is reserved for the two-method extension described in
[`docs/AAAI_RECENT_BASELINES.md`](../../docs/AAAI_RECENT_BASELINES.md).

The extension is intentionally separate from the immutable 16-method
`protocol.json`. Until `protocol_extension.json` reaches `frozen` status, its
models may be used only for implementation tests and official-Tuning studies,
not for Eval reporting.

Planned method identifiers:

- `DPAD_AAAI24`
- `DNE_AAAI25`

Formal completion will require a runner, paper-derived model definitions,
local U/M six-metric smoke evidence, frozen hashes, and controller integration
that writes to a separate results directory.
