# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muliadiaz-hue/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muliadiaz-hue/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A count-model forecaster for GTA VI trophy earn rates on Exophase. It models
players-tracked and per-trophy earned counts separately, each log-linear in
log(1 + hours since launch), shrinks each trophy's earned-count slope toward
its tier mean, and takes the ratio at the target time as the median forecast.
The spread on the logit scale combines a random-walk term, a tracker-refresh
term (one refresh expected about every five hours, each one dragging rates
down as a wave of newer players arrives), and the pooled fit residual. On a
flat history every slope is zero and the forecast collapses to the last read,
the house baseline; the extra machinery only wakes up once the case history
shows real movement. A trophy sitting at exactly 100% keeps all five
quantiles at 100. Python 3.13, standard library only, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=c81nkrnx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
