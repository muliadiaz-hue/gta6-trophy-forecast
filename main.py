"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Idea, in one paragraph: Exophase's earn rates are ratios, earned-count over
players-tracked, and both halves move.  Each refresh of the tracker sweeps in a
wave of newer, more casual players, so the tracked-player count jumps while
the earned counts for a trophy grow only as those players actually play.  This
program therefore models the two counts separately -- log-linear in
log(1 + hours since launch) -- and takes the ratio at the target time as the
median forecast.  Per-trophy earned-count slopes are shrunk toward their tier
mean, because a single trophy's early slope is mostly noise.  The spread is a
random-walk term plus a refresh term (one tracker refresh is expected about
every five hours and each one can drag every rate down) plus the pooled fit
residual, all on the logit scale.  On a flat history every slope is zero and
the forecast collapses to the last read, exactly the house baseline; the extra
machinery only wakes up once the final case's history shows real movement.

Python 3.13, standard library only, deterministic: the output is a pure
function of the case file, trophies are visited in id order, and repeated runs
write identical bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, and the standard-normal quantile at each.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZS = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Launch midnight, the origin of the log-time axis.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread constants, surveyed in prep against the practice proxy:
#   random-walk sigma per sqrt-day, refresh sigma per sqrt-refresh, refresh cadence.
WALK_SIGMA = 0.5
REFRESH_SIGMA = 0.35
REFRESH_EVERY_H = 5.0

LO = 0.01
HI = 99.99


def pct_to_logit(pct):
    """Logit of a percent, clamped first so 0 and 100 never blow up."""
    p = min(HI, max(LO, pct))
    return math.log(p / (100.0 - p))


def logit_to_pct(lg):
    """Inverse logit, back to a percent."""
    return 100.0 / (1.0 + math.exp(-lg))


def round4(value):
    """Round to 4 decimals, half up, the way JavaScript Math.round does it.

    Whole results come back as ints so the JSON writes 100, not 100.0.
    """
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if out == int(out) else out


def read_time(text):
    """Parse an ISO-8601 timestamp, tolerating the trailing Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(moment):
    """log(1 + hours since launch midnight) -- the axis every fit uses."""
    hours = (moment - EPOCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def ols(xs, ys):
    """Least-squares line through (xs, ys): (slope, intercept, resid_sd).

    Returns zeros when the x values carry no spread -- a flat history has no
    trend to learn, and pretending otherwise would invent signal.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    if n <= 2:
        return slope, intercept, 0.0
    resid = math.sqrt(
        sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    )
    return slope, intercept, resid


def earned_points(history, xs, idx):
    """(log-time, log earned-count) points for one trophy, skipping gaps."""
    pts = []
    for read, x in zip(history, xs):
        rate = read["rates"][idx]
        n = read["players_tracked"]
        if rate is None or rate <= 0 or not n or n <= 0:
            continue
        pts.append((x, math.log(rate * n / 100.0)))
    return pts


def last_known_rate(history, idx):
    """Most recent non-null rate for trophy column idx, else None."""
    for read in reversed(history):
        rate = read["rates"][idx]
        if rate is not None:
            return rate
    return None


def median_rates(case):
    """One median forecast rate per trophy, from the count model.

    Falls back to the last read whenever the history is too thin to fit:
    fewer than three reads, a single distinct time axis, or a trophy with no
    usable earned-count series all degrade gracefully to persistence.
    """
    trophies = case["trophies"]
    history = case["history"]
    target_x = log_time(read_time(case["target"]))
    last = history[-1]
    n_last = None
    for read in reversed(history):
        if read["players_tracked"]:
            n_last = read["players_tracked"]
            break
    if n_last is None:
        return medians

    medians = [last_known_rate(history, i) for i in range(len(trophies))]
    medians = [m if m is not None else LO for m in medians]
    if len(history) < 3:
        return medians

    xs = [log_time(read_time(r["at"])) for r in history]
    if len(set(xs)) < 2:
        return medians

    # Tracked players: one log-linear fit over every read.
    ln_n = []
    for read in history:
        n = read["players_tracked"]
        ln_n.append(math.log(n) if n and n > 0 else None)
    usable_n = [(x, y) for x, y in zip(xs, ln_n) if y is not None]
    if len(usable_n) < 2:
        return medians
    slope_n, icept_n, _ = ols([p[0] for p in usable_n], [p[1] for p in usable_n])
    n_target = math.exp(icept_n + slope_n * target_x)
    n_target = min(max(n_target, n_last), 100.0 * n_last)

    # Earned counts per trophy, slopes shrunk toward the tier mean.
    raw = []  # per trophy: (slope, intercept, n_points) or None
    for i in range(len(trophies)):
        pts = earned_points(history, xs, i)
        if len(pts) < 2 or len(set(p[0] for p in pts)) < 2:
            raw.append(None)
        else:
            slope, icept, _ = ols([p[0] for p in pts], [p[1] for p in pts])
            raw.append((slope, icept, len(pts)))

    tier_slopes = {}
    for i, trophy in enumerate(trophies):
        if raw[i] is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(raw[i][0])
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}

    for i in range(len(trophies)):
        if raw[i] is None:
            continue
        slope, icept, npts = raw[i]
        mean = tier_mean.get(trophies[i]["tier"], slope)
        w = npts / (npts + 2.0)  # shrink: few reads -> trust the tier, not the trophy
        slope = w * slope + (1.0 - w) * mean
        e_target = math.exp(icept + slope * target_x)
        medians[i] = min(HI, max(LO, 100.0 * e_target / n_target))
    return medians


def pooled_residual(case):
    """Pooled SD of the log-earned-count fit residuals, across trophies.

    Zero on a flat history (nothing to be wrong about yet); grows once the
    final case's history shows real per-trophy wobble.
    """
    history = case["history"]
    if len(history) < 3:
        return 0.0
    xs = [log_time(read_time(r["at"])) for r in history]
    if len(set(xs)) < 2:
        return 0.0
    sq, dof = 0.0, 0
    for i in range(len(case["trophies"])):
        pts = earned_points(history, xs, i)
        if len(pts) <= 2 or len(set(p[0] for p in pts)) < 2:
            continue
        slope, icept, _ = ols([p[0] for p in pts], [p[1] for p in pts])
        sq += sum((y - (icept + slope * x)) ** 2 for x, y in pts)
        dof += len(pts) - 2
    return math.sqrt(sq / dof) if dof > 0 else 0.0


def spread(case):
    """Forecast spread on the logit scale: walk + refresh + fit residual."""
    horizon_h = case["horizon_minutes"] / 60.0
    walk = WALK_SIGMA * math.sqrt(horizon_h / 24.0)
    refreshes = horizon_h / REFRESH_EVERY_H
    refresh = REFRESH_SIGMA * math.sqrt(refreshes)
    resid = pooled_residual(case)
    return math.sqrt(walk * walk + refresh * refresh + resid * resid)


def forecast_case(case, name):
    if list(case["levels"]) != list(LEVELS):
        raise ValueError(f"{name} asks for levels {case['levels']}, not {list(LEVELS)}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    medians = median_rates(case)
    width = spread(case)
    rows = []
    last = case["history"][-1]
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        if rate == 100:
            # Earned by everyone from the first read and untouched by the
            # first refresh: keep all five quantiles at 100.
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        centre = pct_to_logit(medians[i])
        rows.append(
            {
                "id": trophy["id"],
                "q": [
                    round4(min(HI, max(LO, logit_to_pct(centre + z * width))))
                    for z in ZS
                ],
            }
        )
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
