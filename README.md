# sciqlop-tohban

SciQLop plugin to review BepiColombo/Mio observation plans (`.evt`) during Tohban meetings.

**This repository holds code only.** Observation plans, SPICE kernels and figures are
project data and must stay out of it (`.gitignore` blocks the usual extensions).
Tests use synthetic inline plans.

## Use

Add the repository folder to `~/.config/sciqlop/sciqloppluginssettings.yaml`:

```yaml
extra_plugins_folders:
  - /path/to/sciqlop_tohban
plugins:
  sciqlop_tohban:
    enabled: true
```

Restart SciQLop, then **Tools → Open Tohban observation plan…**.

The `.evt` core (`sciqlop_tohban.evt`) has no GUI dependency and can be used from a
notebook: `parse(text)` / `dump(evt)` round-trip a file byte for byte.

## Tests

```bash
uv run --project /path/to/SciQLop python -m pytest sciqlop_tohban
```
