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

## SPICE views

Once a plan is open, the plugin samples MIO and MPO every 5 minutes over the plan, then adds:

- under the timeline: the expected MIO/MPO region, the distance to Mercury, the
  distance to the MSO X axis next to the magnetopause and bow shock at the same X,
  and the MIO–MPO separation;
- a **Tohban trajectory** panel: MIO in MSO X–Z and in its orbit plane, coloured
  by the highest active operation mode, with MPO, hourly marks and the boundaries;
- a **Tohban MSASI** panel: the field of view in MSO X–Z, and its footprint on a
  10 R_M shell.

The trajectory and MSASI panels redraw the part of the orbit within the timeline's
time range, and follow the mode filters. Boundaries are the Winslow et al. (2013)
average magnetopause and bow shock.

The first time, the plugin downloads ~150 MB of public ESA BepiColombo planning
kernels into `~/.cache/sciqlop_tohban/kernels`. Set `TOHBAN_METAKERNEL` to a
metakernel path to use your own kernel set instead.

The `.evt` core (`sciqlop_tohban.evt`) has no GUI dependency and can be used from a
notebook: `parse(text)` / `dump(evt)` round-trip a file byte for byte.

## Credits

The toolbar icon is the BepiColombo mission patch, © ESA/JAXA, from
<https://www.cosmos.esa.int/web/bepicolombo>.

## Tests

```bash
uv run --project /path/to/SciQLop python -m pytest sciqlop_tohban
```
