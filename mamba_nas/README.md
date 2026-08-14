# MambaTSC-NSGA2 module

This directory is an independent classification NAS pipeline. It never imports or modifies the
original `run.py`, `models/`, `exp/`, or `data_provider/` training pipeline.

The production model imports `mamba_ssm.Mamba`. CPU tests inject a small deterministic mixer only
to verify architecture, masking, search operators, data isolation, and exports; that mixer is not a
search backend.

The search stage reads only `{dataset}_TRAIN.ts`. The explicitly named
`load_test_for_refit` function is called only after representative architectures have been selected.

