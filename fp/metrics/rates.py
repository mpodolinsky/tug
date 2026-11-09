"""Aggregate outcomes into rates. All values are percentages in [0, 100]; NaN when the denominator is empty."""

from dataclasses import dataclass

from .outcomes import Outcome

NAN = float("nan")


@dataclass(frozen=True)
class Rates:
    tpr: float  # (failures - missed) / failures
    tpr_early: float  # (early + on_time) / failures: alarm at or before the onset
    tpr_on_time: float  # on_time / failures
    ldr: float  # late / (early + late); NaN if no alarm fell outside the window
    fpr: float  # false alarms / successes


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else NAN


def rates_from_outcomes(outcomes: list[Outcome]) -> Rates:
    n = {o: 0 for o in Outcome}
    for o in outcomes:
        n[o] += 1
    failures = n[Outcome.EARLY] + n[Outcome.ON_TIME] + n[Outcome.LATE] + n[Outcome.MISSED]
    successes = n[Outcome.FALSE_ALARM] + n[Outcome.CORRECT_REJECT]
    off_window = n[Outcome.EARLY] + n[Outcome.LATE]
    return Rates(
        tpr=_pct(failures - n[Outcome.MISSED], failures),
        tpr_early=_pct(n[Outcome.EARLY] + n[Outcome.ON_TIME], failures),
        tpr_on_time=_pct(n[Outcome.ON_TIME], failures),
        ldr=_pct(n[Outcome.LATE], off_window),
        fpr=_pct(n[Outcome.FALSE_ALARM], successes),
    )
