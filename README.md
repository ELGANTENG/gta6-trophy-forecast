# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ELGANTENG/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ELGANTENG/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy. Built for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty.

## How it works

`main.ts` (Node 22, standard library only, no network) reads every case file in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `/output/<id>.out` per case. The model, per trophy, works on the logit scale:

- **Trend, split in two.** A rate is earners over players tracked, and the two sides move differently: earners grow as players play, while the tracked base jumps at Exophase refreshes and then saturates. So the trend fits two least-squares lines against log-time — log(earners) for the trophy and log(players tracked) for the game — over the last 24 hours of reads, reads the earners line at the target, and divides by the players line damped for saturation.
- **Refresh risk.** Exophase refreshes on its own schedule: on launch night the page sat frozen (638 players, every rate identical) for hours, then refreshed to 3,868 players and every rate but the one at 100% lower. When the page is frozen at the checkpoint, the forecast is a two-component mixture: no refresh before the target (rates follow the trend) versus a refresh (rates diluted by the expected jump in players tracked), with the refresh probability from a Weibull hazard on the freeze length.
- **Spread.** A random-walk term that widens with the horizon, plus the trend fit's residual spread, plus extra spread for the rarest trophies. The trophy at 100% is forecast at 100% flat.

On the practice cases the model beats the house baseline under every simulated refresh (rates scaled ×0.15–×0.8): at the refresh the house's own stated baseline loss (~1.51) pins down (×0.2), it scores ~0.90–0.93 against the baseline's ~1.50.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Deterministic: two runs write byte-identical output. Runs in well under a second per case.

---

Entered on [make.fun](https://make.fun/?ref=qd0ywjzn&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
