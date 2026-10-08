"""Electricity used by this session, measured on the local machine.

Apple Silicon keeps cumulative energy counters for its CPU, GPU, Neural
Engine and DRAM — the numbers `powermetrics` reports, readable without sudo
through the private IOReport library. Reading them at the start and end of a
block gives that block's energy exactly, with no sampling thread.
`session_meter()` takes the first reading when it's first called; wrap each
crew invocation in `measure()` to get its own total. Time outside any
`measure()` block counts as idle, and its average draw is the baseline that
"above idle" subtracts.

With a local model backend (e.g. Ollama, which runs on the GPU) the
above-idle figure is mostly inference. Remote inference (Gemini) and Earth
Engine compute run on Google's servers and don't show up here; neither do
the display, SSD, Wi-Fi or charger losses — this is the chip and its memory.
Elsewhere (Intel Macs, Linux) the meter reports `available == False`.

Usage:
    from natcap_agents import energy
    meter = energy.session_meter()
    with meter.measure("forest loss") as run:
        crew.run("How much forest was lost in <region> since 2015?")
    print(run.energy_wh, run.above_idle_wh(meter.idle_w), meter.session_wh)
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

# Per-block totals in IOReport's "Energy Model" group; the other ~300 channels
# (per-core, SRAM, fabric, ...) are components of these or unrelated blocks.
_CHANNELS = {"CPU Energy", "GPU Energy", "ANE", "DRAM"}
_JOULES_PER_UNIT = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9}
_MIN_IDLE_SECONDS = 10.0  # don't report an idle baseline from a moment's gap
_UTF8 = 0x08000100


class _IOReport:
    """Minimal ctypes binding to libIOReport's Energy Model counters."""

    def __init__(self):
        vp = ctypes.c_void_p
        self._cf = cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        self._ior = ior = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        for fn, restype, argtypes in (
            (cf.CFStringCreateWithCString, vp, [vp, ctypes.c_char_p, ctypes.c_uint32]),
            (cf.CFStringGetCString, ctypes.c_bool, [vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
            (cf.CFDictionaryCreateMutableCopy, vp, [vp, ctypes.c_long, vp]),
            (cf.CFDictionaryGetValue, vp, [vp, vp]),
            (cf.CFArrayGetCount, ctypes.c_long, [vp]),
            (cf.CFArrayGetValueAtIndex, vp, [vp, ctypes.c_long]),
            (cf.CFRelease, None, [vp]),
            (ior.IOReportCopyChannelsInGroup, vp, [vp, vp, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]),
            (ior.IOReportCreateSubscription, vp, [vp, vp, ctypes.POINTER(vp), ctypes.c_uint64, vp]),
            (ior.IOReportCreateSamples, vp, [vp, vp, vp]),
            (ior.IOReportChannelGetChannelName, vp, [vp]),
            (ior.IOReportChannelGetUnitLabel, vp, [vp]),
            (ior.IOReportSimpleGetIntegerValue, ctypes.c_int64, [vp, ctypes.c_int32]),
        ):
            fn.restype, fn.argtypes = restype, argtypes

        group = cf.CFStringCreateWithCString(None, b"Energy Model", _UTF8)
        channels = ior.IOReportCopyChannelsInGroup(group, None, 0, 0, 0)
        cf.CFRelease(group)
        if not channels:
            raise OSError("no IOReport Energy Model channels on this machine")
        self._wanted = cf.CFDictionaryCreateMutableCopy(None, 0, channels)
        cf.CFRelease(channels)
        self._subscribed = vp()
        self._subscription = ior.IOReportCreateSubscription(
            None, self._wanted, ctypes.byref(self._subscribed), 0, None)
        if not self._subscription:
            raise OSError("IOReport subscription failed")
        self._channels_key = cf.CFStringCreateWithCString(None, b"IOReportChannels", _UTF8)
        self._buf = ctypes.create_string_buffer(128)

    def _str(self, ref) -> str | None:
        if ref and self._cf.CFStringGetCString(ref, self._buf, len(self._buf), _UTF8):
            return self._buf.value.decode()
        return None

    def energy_j(self) -> float:
        cf, ior = self._cf, self._ior
        samples = ior.IOReportCreateSamples(self._subscription, self._subscribed, None)
        if not samples:
            raise OSError("IOReport sample failed")
        try:
            channels = cf.CFDictionaryGetValue(samples, self._channels_key)
            if not channels:
                raise OSError("IOReport sample has no channels")
            total = 0.0
            for i in range(cf.CFArrayGetCount(channels)):
                channel = cf.CFArrayGetValueAtIndex(channels, i)
                if self._str(ior.IOReportChannelGetChannelName(channel)) not in _CHANNELS:
                    continue
                scale = _JOULES_PER_UNIT.get(self._str(ior.IOReportChannelGetUnitLabel(channel)))
                if scale:
                    total += ior.IOReportSimpleGetIntegerValue(channel, 0) * scale
            return total
        finally:
            cf.CFRelease(samples)


@functools.cache
def _reader() -> _IOReport | None:
    if sys.platform != "darwin":
        return None
    try:
        return _IOReport()
    except (OSError, AttributeError):  # AttributeError: a symbol missing from the dylib
        return None


def soc_energy_j() -> float | None:
    """Joules the chip's CPU, GPU, Neural Engine and DRAM have used since boot,
    or None where those counters aren't available."""
    reader = _reader()
    if reader is None:
        return None
    try:
        return reader.energy_j()
    except OSError:
        return None


@dataclass
class Run:
    label: str
    seconds: float = 0.0
    joules: float = 0.0

    @property
    def energy_wh(self) -> float:
        return self.joules / 3600

    @property
    def avg_w(self) -> float | None:
        return self.joules / self.seconds if self.seconds else None

    def above_idle_wh(self, idle_w: float | None) -> float | None:
        """Energy beyond what the chip would have drawn idling for as long."""
        if idle_w is None:
            return None
        return (self.joules - idle_w * self.seconds) / 3600


class EnergyMeter:
    def __init__(self, read_energy: Callable[[], float | None] = soc_energy_j):
        self._read = read_energy
        self.started_at = time.monotonic()
        self._start_j = read_energy()
        self.available = self._start_j is not None
        self.runs: list[Run] = []

    @contextmanager
    def measure(self, label: str) -> Iterator[Run]:
        """Attribute the energy used inside this block to a new Run."""
        run = Run(label)
        start_t, start_j = time.monotonic(), self._read()
        try:
            yield run
        finally:
            end_j = self._read()
            run.seconds = time.monotonic() - start_t
            if start_j is not None and end_j is not None:
                run.joules = end_j - start_j
            self.runs.append(run)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def session_joules(self) -> float:
        now = self._read() if self.available else None
        return now - self._start_j if now is not None else 0.0

    @property
    def session_wh(self) -> float:
        return self.session_joules / 3600

    @property
    def idle_w(self) -> float | None:
        """Average draw outside measure() blocks, once there's enough of it.
        Read it between runs: inside a block, the current run counts as idle."""
        idle_s = self.elapsed_s - sum(r.seconds for r in self.runs)
        if idle_s < _MIN_IDLE_SECONDS:
            return None
        return (self.session_joules - sum(r.joules for r in self.runs)) / idle_s


_meter: EnergyMeter | None = None


def session_meter() -> EnergyMeter:
    """The process-wide meter, started on first call (like results' one board)."""
    global _meter
    if _meter is None:
        _meter = EnergyMeter()
    return _meter
