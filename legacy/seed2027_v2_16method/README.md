# Legacy 16-method protocol

This directory preserves the protocol that was committed at Git revision
`ff501df` before KNN was removed from the active comparison on 2026-07-21.
It exists for auditability only. New runs, summaries, and paper tables must use
the repository-root `protocol.json`, which contains 15 core methods.

The old protocol had 16 methods x 193 series = 3,088 units. Its historical
controller SHA256 was
`88a394978d2547608e2ee58e30e9797a58a6bc950ca950c8c802a8b567dd8d78`.
