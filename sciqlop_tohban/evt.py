"""Lossless reader/writer for observation-plan ``.evt`` files.

A file is a comment header followed by ``COMMAND YYYY-MM-DDTHH:MM:SS`` lines.
Parsing is strict so that ``dump(parse(text)) == text`` always holds.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np

_RECORD = re.compile(r"^(\S+) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)$")
_TIME_UNIT = "s"
_SUBMODES = {"HKM", "LM", "MM"}


class Record(NamedTuple):
    command: str
    time: np.datetime64


class Command(NamedTuple):
    instrument: str
    mode: str
    action: str | None


@dataclass(frozen=True)
class Interval:
    instrument: str
    mode: str
    start: np.datetime64
    stop: np.datetime64
    open_ended: bool = False


@dataclass(frozen=True)
class EvtFile:
    header: tuple[str, ...]
    records: tuple[Record, ...]
    newline: str = "\n"
    final_newline: bool = True


def _parse_record(line: str, number: int) -> Record:
    match = _RECORD.match(line)
    if match is None:
        raise ValueError(f"line {number}: expected 'COMMAND YYYY-MM-DDTHH:MM:SS', got {line!r}")
    return Record(match[1], np.datetime64(match[2], _TIME_UNIT))


def _is_header(line: str) -> bool:
    return line.startswith("#") or not line.strip()


def parse(text: str) -> EvtFile:
    newline = "\r\n" if "\r\n" in text else "\n"
    final_newline = text.endswith(newline)
    lines = text.split(newline)
    if final_newline:
        lines.pop()
    header_size = next((i for i, line in enumerate(lines) if not _is_header(line)), len(lines))
    records = tuple(
        _parse_record(line, number)
        for number, line in enumerate(lines[header_size:], start=header_size + 1)
    )
    return EvtFile(tuple(lines[:header_size]), records, newline, final_newline)


def format_record(record: Record) -> str:
    return f"{record.command} {np.datetime_as_string(record.time.astype(f'datetime64[{_TIME_UNIT}]'))}"


def dump(evt: EvtFile) -> str:
    lines = [*evt.header, *map(format_record, evt.records)]
    return evt.newline.join(lines) + (evt.newline if evt.final_newline else "")


def parse_command(name: str) -> Command:
    stem = name.removeprefix("SI_")
    action = next((a for a in ("ON", "OFF") if stem.endswith(f"_{a}")), None)
    if action:
        stem = stem.removesuffix(f"_{action}")
    if "_OBS_MODE_" in stem:
        instrument, mode = stem.split("_OBS_MODE_", 1)
        return Command(instrument, f"OBS:{mode}", action)
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1] == "HKM" and parts[-2] in {"L", "M"}:
        return Command("_".join(parts[:-2]), f"{parts[-2]}_HKM", action)
    if len(parts) >= 2 and parts[-1] in _SUBMODES:
        return Command("_".join(parts[:-1]), parts[-1], action)
    return Command(stem, "BASE", action)


def pair_intervals(records: Iterable[Record]) -> tuple[list[Interval], list[Record]]:
    """Pair ON/OFF commands into intervals; commands without ON/OFF are points.

    Pairing is FIFO per command stem because some plans hold overlapping ON
    windows for the same command. An ON never closed ends at the last record.
    An OFF with no pending ON is dropped.
    """
    records = list(records)
    pending: dict[str, deque[Record]] = defaultdict(deque)
    intervals: list[Interval] = []
    points: list[Record] = []
    for record in records:
        command = parse_command(record.command)
        if command.action is None:
            points.append(record)
            continue
        stem = record.command.removesuffix(f"_{command.action}")
        if command.action == "ON":
            pending[stem].append(record)
        elif pending[stem]:
            start = pending[stem].popleft()
            intervals.append(Interval(command.instrument, command.mode, start.time, record.time))
    last = records[-1].time if records else None
    intervals += [
        Interval(*parse_command(on.command)[:2], on.time, last, open_ended=True)
        for queue in pending.values()
        for on in queue
    ]
    return sorted(intervals, key=lambda i: i.start), points


def _power_windows(intervals: Iterable[Interval]) -> dict[str, list[Interval]]:
    windows: dict[str, list[Interval]] = defaultdict(list)
    for interval in sorted(intervals, key=lambda i: i.start):
        if interval.mode == "BASE":
            windows[interval.instrument].append(interval)
    return windows


def _switches(points: Iterable[Record]) -> dict[str, list[tuple[np.datetime64, str]]]:
    switches: dict[str, list[tuple[np.datetime64, str]]] = defaultdict(list)
    for record in points:
        command = parse_command(record.command)
        if command.mode.startswith("OBS:"):
            switches[command.instrument].append((record.time, command.mode))
    return {instrument: sorted(items) for instrument, items in switches.items()}


def _enclosing(windows: list[Interval], starts: list[np.datetime64], time: np.datetime64) -> Interval | None:
    # simplify: takes the latest-started window only; overlapping power windows
    # (seen on DPU only, which has no OBS modes) could need a scan of earlier ones.
    index = bisect_right(starts, time) - 1
    return windows[index] if index >= 0 and time < windows[index].stop else None


def obs_mode_intervals(points: Iterable[Record], intervals: Iterable[Interval]) -> list[Interval]:
    """Turn OBS_MODE switch commands into states.

    A mode holds until the next switch of the same instrument or until the
    instrument is powered off. A switch outside any power window is dropped.
    """
    windows = _power_windows(intervals)
    states = []
    for instrument, switches in _switches(points).items():
        own = windows.get(instrument, [])
        starts = [w.start for w in own]
        next_times = [time for time, _ in switches[1:]] + [None]
        for (time, mode), next_time in zip(switches, next_times):
            window = _enclosing(own, starts, time)
            if window is not None:
                stop = window.stop if next_time is None else min(next_time, window.stop)
                states.append(Interval(instrument, mode, time, stop))
    return sorted(states, key=lambda i: i.start)

