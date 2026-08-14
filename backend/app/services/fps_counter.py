"""
Real-time FPS (Frames Per Second) Counter Utility.
Uses high-resolution timer and exponential moving average to measure pipeline processing speed.
"""
import time
from collections import deque

class FPSCounter:
    """
    Measures frame rate (FPS) over a rolling window.
    """
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_timestamps = deque(maxlen=window_size)
        self.start_time = None
        self.total_frames = 0

    def start(self):
        """Starts the timer."""
        self.start_time = time.perf_counter()
        self.frame_timestamps.clear()
        self.total_frames = 0

    def update(self) -> float:
        """
        Call at every frame render/process step.
        Returns the current rolling average FPS.
        """
        now = time.perf_counter()
        if self.start_time is None:
            self.start_time = now

        self.total_frames += 1
        self.frame_timestamps.append(now)

        if len(self.frame_timestamps) < 2:
            return 0.0

        # Calculate FPS based on timestamps in window
        elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if elapsed <= 0:
            return 0.0

        fps = (len(self.frame_timestamps) - 1) / elapsed
        return round(fps, 1)

    def get_avg_fps(self) -> float:
        """Returns overall average FPS since start."""
        if self.start_time is None or self.total_frames == 0:
            return 0.0
        elapsed = time.perf_counter() - self.start_time
        if elapsed <= 0:
            return 0.0
        return round(self.total_frames / elapsed, 1)
