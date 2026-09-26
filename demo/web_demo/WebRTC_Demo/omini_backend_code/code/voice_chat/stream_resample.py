"""Stateful resampler for streamed int16 mono audio (sglang-omni bridge latency patch).

scipy's resample_poly filters each call on its own, so cutting a stream into 80 ms pieces adds an edge
transient at every cut. This keeps the FIR state across calls instead. The filter is the one resample_poly
designs (Kaiser beta 5, half length 10 * max(up, down)); being causal, it delays the signal by half the
filter length (0.6 ms for 48 -> 16 kHz, 0.4 ms for 24 -> 48 kHz).
"""
from math import gcd

import numpy as np
from scipy.signal import firwin, lfilter


class StreamResampler:
    def __init__(self, source_rate: int, target_rate: int):
        g = gcd(source_rate, target_rate)
        self.up, self.down = target_rate // g, source_rate // g
        self.identity = self.up == self.down == 1
        if not self.identity:
            max_rate = max(self.up, self.down)
            self.taps = firwin(2 * 10 * max_rate + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * self.up
        self.reset()

    def reset(self) -> None:
        self.state = None if self.identity else np.zeros(len(self.taps) - 1)
        self.offset = 0  # position of the next input sample in the upsampled stream, modulo down

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self.identity or len(audio) == 0:
            return np.asarray(audio, dtype=np.int16)
        x = np.asarray(audio, dtype=np.float64)
        if self.up > 1:
            stuffed = np.zeros(len(x) * self.up)
            stuffed[:: self.up] = x
            x = stuffed
        y, self.state = lfilter(self.taps, 1.0, x, zi=self.state)
        start = (-self.offset) % self.down
        self.offset = (self.offset + len(x)) % self.down
        return np.clip(np.rint(y[start :: self.down]), -32768, 32767).astype(np.int16)
