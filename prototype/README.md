# prototype

Implementation of the tyre pace-loss and stint-survival pipeline.

```bash
uv run main.py            # full pipeline
uv run main.py --list     # show the stages
uv run main.py --only recommend
```

`main.py` holds the stage list and the ordering. Every other `.py` file is one
stage: it reads from `data/`, writes back to `data/`, and can be run on its own
with `uv run <file>.py`.

See the [root README](../README.md) for prerequisites and the full stage table.
