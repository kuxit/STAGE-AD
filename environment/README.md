# Environments

The local and formal environments have different purposes.

- `requirements-local-smoke.txt` records the existing Windows `00gwk`
  environment used to prove that every method can execute and emit six finite
  metrics on real TSB-AD data. Its runtime is never paper evidence.
- `requirements-formal.txt` records the common Python packages from Server B.
  The original formal run used Python 3.10, PyTorch 2.8.0 + CUDA 12.8 for most
  methods, and PyTorch 2.7.1 + CUDA 12.8 for MEMTO. Those two PyTorch installs
  must be recreated separately on the server.

A local smoke pass proves interface and dependency completeness. It does not
authorize changing any frozen method, configuration, or metric implementation.
