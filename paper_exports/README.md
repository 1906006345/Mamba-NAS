# Paper exports

`python -m mamba_nas.cli export-paper --run <run_id>` writes lightweight, regenerable CSV and JSON
tables here. Candidate checkpoints and per-sample predictions remain under the ignored `runs/`
directory and are never copied into paper exports.

