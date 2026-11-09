"""Classify one episode given its scores and a per-step threshold. The only place that does this.

Alarm step = first step with score >= threshold (None if never).
For a failure with onset o and on-time window w, with delay = alarm - o:

    early    delay < -w          (before the window)
    on_time  -w <= delay <= 0    (inside the window, onset included)
    late     delay > 0
    missed   no alarm

``tpr_early`` everywhere means early + on_time, i.e. an alarm at or before the onset.
For a success any alarm is a false alarm.
"""

from enum import Enum

import numpy as np


class Outcome(str, Enum):
    EARLY = "early"
    ON_TIME = "on_time"
    LATE = "late"
    MISSED = "missed"
    FALSE_ALARM = "false_alarm"
    CORRECT_REJECT = "correct_reject"


def alarm_step(scores: np.ndarray, threshold: np.ndarray) -> int | None:
    """First step with score >= threshold, or None. The threshold must cover every step."""
    if len(threshold) < len(scores):
        raise ValueError(f"threshold has {len(threshold)} steps, scores have {len(scores)}")
    hit = np.flatnonzero(scores >= threshold[: len(scores)])
    return int(hit[0]) if len(hit) else None


def classify(scores: np.ndarray, threshold: np.ndarray, onset: int | None, window: int) -> Outcome:
    """``onset`` is None for a success."""
    alarm = alarm_step(scores, threshold)
    if onset is None:
        return Outcome.CORRECT_REJECT if alarm is None else Outcome.FALSE_ALARM
    if alarm is None:
        return Outcome.MISSED
    delay = alarm - onset
    if delay < -window:
        return Outcome.EARLY
    return Outcome.ON_TIME if delay <= 0 else Outcome.LATE
