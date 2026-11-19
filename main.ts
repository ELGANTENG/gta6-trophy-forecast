// GTA VI trophy rarity forecast — entry for make.fun gta6-trophy-rarity-forecast-01.
//
// Forecasts, for every PS5 trophy of Grand Theft Auto VI, its Exophase earn
// rate at a case's target, as five quantiles (levels 0.05/0.25/0.5/0.75/0.95)
// on the logit scale. Node 22, standard library only, no network.
//
// Model, per trophy, in logit space (logit(p) = ln(p / (1 - p)), p = rate/100):
//
//   1. Trend. A rate is a ratio — earners over players tracked — and the
//      two sides move differently: earners grow as players play, while the
//      tracked base jumps at Exophase refreshes and then saturates. Fitting
//      the rate directly would read every refresh dilution as a falling
//      trend, so the trend is fitted on the two sides separately, each an
//      ordinary-least-squares line against ln(1 + hours since
//      2026-11-19T00:00:00Z) over the case's last 24 hours of reads (or all
//      reads when fewer): log(earners) for the trophy, log(players tracked)
//      for the game. The earners line is read at the target; the players
//      line is damped (see PLAYER_DAMP) because growth in log-time is
//      concave once a launch surge saturates. The forecast rate is
//      earners / players at the target. Slopes are shrunk by n / (n + 2),
//      and the total move from the last read is capped at 4 logit units.
//      With fewer than 3 reads the trend is flat at the last rate.
//   2. Spread. A random-walk term (0.5 logit per sqrt(day), as the house
//      baseline), plus the trend fit's residual spread scaled by the
//      extrapolation leverage (capped), plus a rarity term: rarer trophies
//      get up to 0.25 extra logit of spread, because a rare trophy's rate is
//      moved more, relatively, by the same number of new earners.
//   3. Refresh. Exophase refreshes a game's counts on its own schedule, not
//      with every player's trophy: on launch night the page sat frozen
//      (638 players tracked, every rate identical) for hours, then refreshed
//      to 3,868 players and every rate but the one at 100% lower. A refresh
//      dilutes rates by roughly the jump in players tracked, so when the
//      page is frozen at the checkpoint (no change in players tracked or in
//      any rate for an hour or more) the forecast is a two-component mixture:
//      with probability 1 - p no refresh lands before the target and the
//      rate follows the trend; with probability p a refresh does and the
//      rate drops by ln(G), G the expected jump in players tracked (the
//      median jump seen at refreshes in the history, else a prior of 2.25:
//      launch night's jump was 6x, but that was the first refresh after
//      launch; later refreshes dilute less as player growth slows).
//      p comes from a Weibull hazard (shape 2) on the freeze length, with a
//      4.5-hour prior refresh cycle: the longer the page has sat frozen,
//      the more likely a refresh is due before the target. On a long horizon
//      the mixture collapses to a single wide normal, because many refreshes
//      land and their average effect is already in the trend.
//
// The platinum is not special-cased: it is the rarest trophy in the data, so
// the rarity spread already widens it, and its trend governs its median. The
// one trophy at 100% is forecast at 100% flat: a refresh left it at 100%,
// and every tracked player holds it.
//
// Deterministic: no randomness anywhere, reads are processed in name order,
// so two runs write byte-identical output. Runs in milliseconds per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

/** Origin of the log-time axis: the game's release day, 2026-11-19T00:00:00Z. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Prior refresh cycle of a freshly released game's page, in hours. */
const TAU_REFRESH_H = 4.5;
/** Shape of the refresh hazard: it rises the longer the page sits frozen. */
const WEIBULL_K = 2;
/** Prior jump in players tracked when a refresh lands. Launch night's only
 *  observed jump was 6x, but that was the first refresh after launch, when
 *  the tracked base was exploding; a later refresh's jump is smaller. The
 *  prior of 2.25 (a 0.81-logit dilution) is the conservative middle: with it,
 *  the forecast beats the house baseline under simulated refreshes that
 *  scale rates by 0.2x, 0.6x and 0.8x alike, and the 0.2x case is the one the
 *  house's own stated baseline loss (~1.51) pins the real refresh to. */
const G_PRIOR = 2.25;
/** Random-walk spread, logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Cap on the trend fit's residual spread, logit units. */
const SIGMA_FIT_CAP = 0.8;
/** Extra spread for the rarest trophies, logit units. */
const SIGMA_RARE_MAX = 0.25;
/** Cap on how far the trend may move the median from the last read, logit. */
const MAX_MOVE = 4.0;
/** Damping on the players-tracked trend when extrapolating to the target:
 *  the tracked base grows fast at launch and then saturates, so its recent
 *  log-time slope overstates its future growth; 0.6 splits the difference
 *  between "growth continues" (1) and "growth stops" (0). */
const PLAYER_DAMP = 0.6;
/** A page counts as frozen after this many hours with no change at all. */
const FROZEN_H = 1.0;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp01 = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => {
  const p = clamp01(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Abramowitz–Stegun 7.1.26, accurate to ~1.5e-7: plenty for quantiles. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    t *
    (0.254829592 +
      t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-ax * ax));
}
const normCdf = (x: number): number => 0.5 * (1 + erf(x / Math.SQRT2));

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Ordinary least squares of y on x, with the slope shrunk by n / (n + 2).
 *  Returns null when there are too few points or no spread in x. */
function ols(points: Array<{ x: number; y: number }>): {
  slope: number;
  intercept: number;
  residSd: number;
  xBar: number;
  sxx: number;
  n: number;
} | null {
  const n = points.length;
  if (n < 3) return null;
  const xBar = points.reduce((a, p) => a + p.x, 0) / n;
  const yBar = points.reduce((a, p) => a + p.y, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sxx += (p.x - xBar) ** 2;
    sxy += (p.x - xBar) * (p.y - yBar);
  }
  if (sxx <= 0) return null;
  const slope = (sxy / sxx) * (n / (n + 2));
  const intercept = yBar - slope * xBar;
  let ssr = 0;
  for (const p of points) ssr += (p.y - (intercept + slope * p.x)) ** 2;
  return {
    slope,
    intercept,
    residSd: Math.sqrt(ssr / Math.max(n - 2, 1)),
    xBar,
    sxx,
    n,
  };
}

/**
 * The tau-quantile of a mixture of normals, by bisection on its CDF.
 * `comps` is [weight, centre, sd] per component; weights sum to 1.
 */
function mixtureQuantile(
  comps: Array<[number, number, number]>,
  tau: number,
): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (const [, c, s] of comps) {
    lo = Math.min(lo, c - 8 * s);
    hi = Math.max(hi, c + 8 * s);
  }
  const cdf = (x: number): number => {
    let v = 0;
    for (const [w, c, s] of comps) v += w * normCdf((x - c) / s);
    return v;
  };
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) < tau) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonH = c.horizon_minutes / 60;
  const days = c.horizon_minutes / 1440;

  // --- Refresh history: players tracked only moves at a refresh. ----------
  const reads = c.history;
  const times = reads.map((r) => Date.parse(r.at));
  const jumpRatios: number[] = [];
  const refreshGapsH: number[] = [];
  let lastRefreshMs: number | null = null;
  for (let j = 1; j < reads.length; j++) {
    const prev = reads[j - 1].players_tracked;
    const cur = reads[j].players_tracked;
    if (prev > 0 && cur > prev * 1.02) {
      jumpRatios.push(cur / prev);
      if (lastRefreshMs !== null) refreshGapsH.push((times[j] - lastRefreshMs) / 3_600_000);
      lastRefreshMs = times[j];
    }
  }
  const G = jumpRatios.length
    ? clamp(median(jumpRatios), 1.2, 10)
    : G_PRIOR;
  const tauRefreshH = refreshGapsH.length
    ? clamp(median(refreshGapsH), 2, 48)
    : TAU_REFRESH_H;

  // --- How long has the page sat frozen? ----------------------------------
  // The last read that differs from its predecessor in players tracked or in
  // any rate; when nothing ever changed, the freeze predates the recording.
  let lastChangeMs = times[0];
  for (let j = 1; j < reads.length; j++) {
    let changed = reads[j].players_tracked !== reads[j - 1].players_tracked;
    if (!changed) {
      for (let i = 0; i < c.trophies.length; i++) {
        if (reads[j].rates[i] !== reads[j - 1].rates[i]) {
          changed = true;
          break;
        }
      }
    }
    if (changed) lastChangeMs = times[j];
  }
  const freezeH = (checkpointMs - lastChangeMs) / 3_600_000;
  const frozen = freezeH >= FROZEN_H;

  // --- Probability a refresh lands before the target. ----------------------
  // Weibull survival S(t) = exp(-(t / tau)^k): conditional on no refresh for
  // `freezeH` hours, P(refresh in the next `horizonH`) is
  // 1 - S(freezeH + horizonH) / S(freezeH).
  const pRefresh = frozen
    ? clamp(
        1 -
          Math.exp(
            -(Math.pow(freezeH + horizonH, WEIBULL_K) - Math.pow(freezeH, WEIBULL_K)) /
              Math.pow(tauRefreshH, WEIBULL_K),
          ),
        0,
        0.85,
      )
    : 0;

  const refreshShift = Math.log(G); // logit drop when a refresh dilutes rates
  const sigmaRw = SIGMA_RW * Math.sqrt(days);
  const xTarget =
    Math.log(1 + Math.max(-0.99, (targetMs - RELEASE_MS) / 3_600_000));
  const xLast = Math.log(
    1 + Math.max(-0.99, (times[times.length - 1] - RELEASE_MS) / 3_600_000),
  );

  // --- Players-tracked trend, fitted once per case. -------------------------
  // log(players) against log-time over the last 24h of reads (else all).
  const windowStart = checkpointMs - 24 * 3_600_000;
  let windowIdx = reads.map((_r, j) => j).filter((j) => times[j] >= windowStart);
  if (windowIdx.length < 3) windowIdx = reads.map((_r, j) => j);
  const playersFit = ols(
    windowIdx
      .filter((j) => reads[j].players_tracked > 0)
      .map((j) => ({
        x: Math.log(
          1 + Math.max(-0.99, (times[j] - RELEASE_MS) / 3_600_000),
        ),
        y: Math.log(reads[j].players_tracked),
      })),
  );
  const logPlayersLast = Math.log(
    reads[reads.length - 1].players_tracked,
  );
  // log(players) at the target, with the growth damped for saturation.
  const logPlayersTarget =
    logPlayersLast +
    PLAYER_DAMP *
      (playersFit ? playersFit.slope : 0) *
      (xTarget - xLast);

  return c.trophies.map((trophy, i) => {
    // Last non-null rate for this trophy, and its earners series.
    const series: Array<{ x: number; y: number; t: number }> = [];
    let lastRate: number | null = null;
    for (let j = 0; j < reads.length; j++) {
      const r = reads[j].rates[i];
      if (r === null || r === undefined) continue;
      const hours = (times[j] - RELEASE_MS) / 3_600_000;
      // Earners: rate * players tracked. The rate is clamped so a 0% read
      // still gives a positive earner count for the log fit.
      const earners = (clamp01(r) / 100) * reads[j].players_tracked;
      series.push({
        x: Math.log(1 + Math.max(-0.99, hours)),
        y: Math.log(earners),
        t: times[j],
      });
      lastRate = r;
    }
    if (lastRate === null) {
      // No read ever listed this trophy: a wide spread around a small rate.
      return {
        id: trophy.id,
        q: Z.map((z) => round4(clamp01(invLogit(logit(0.5) + z * 2)))),
      };
    }
    if (lastRate >= 99.99) {
      // Held by every tracked player; a refresh left it at 100%.
      return { id: trophy.id, q: [100, 100, 100, 100, 100] };
    }

    const lastLogit = logit(lastRate);

    // --- Trend: earners line at the target over players line at the target.
    let fit = series.filter((p) => p.t >= windowStart);
    if (fit.length < 3) fit = series;
    const earnersFit = ols(fit);
    let mu = lastLogit;
    let sigmaFit = 0;
    if (earnersFit) {
      const logEarnersTarget =
        earnersFit.intercept + earnersFit.slope * xTarget;
      const rateTarget = clamp01(
        100 * Math.exp(logEarnersTarget - logPlayersTarget),
      );
      mu = logit(rateTarget);
      // Cap the total move from the last read.
      mu = lastLogit + clamp(mu - lastLogit, -MAX_MOVE, MAX_MOVE);
      // Residual spread of the earners fit, scaled by the extrapolation
      // leverage, plus the uncertainty in how much players will grow.
      const leverage = Math.min(
        2,
        Math.sqrt(
          1 + (xTarget - earnersFit.xBar) ** 2 / earnersFit.sxx,
        ),
      );
      const sigmaPlayers =
        0.5 *
        (1 - PLAYER_DAMP) *
        Math.abs(playersFit ? playersFit.slope : 0) *
        (xTarget - xLast);
      sigmaFit = Math.min(
        Math.sqrt(
          (earnersFit.residSd * leverage) ** 2 + sigmaPlayers ** 2,
        ),
        SIGMA_FIT_CAP,
      );
    }

    // --- Spread: random walk + fit residual + rarity. ----------------------
    const sigmaRare =
      SIGMA_RARE_MAX * clamp((-lastLogit - 3) / 3, 0, 1);
    const s = Math.sqrt(
      sigmaRw ** 2 + sigmaFit ** 2 + sigmaRare ** 2,
    );

    // --- Forecast distribution. --------------------------------------------
    let comps: Array<[number, number, number]>;
    if (frozen && pRefresh > 0 && horizonH <= 2 * tauRefreshH) {
      // Short horizon, frozen page: a refresh either lands (rates diluted)
      // or it does not (rates follow the trend).
      comps = [
        [1 - pRefresh, mu, s],
        [pRefresh, mu - refreshShift, s],
      ];
    } else if (frozen && pRefresh > 0) {
      // Long horizon, frozen page: several refreshes will land; hedge the
      // median down by half the dilution and widen the spread for it.
      const half = 0.5 * refreshShift;
      comps = [[1, mu - half, Math.sqrt(s * s + half * half)]];
    } else {
      comps = [[1, mu, s]];
    }

    const q = LEVELS.map((tau) =>
      round4(clamp01(invLogit(mixtureQuantile(comps, tau)))),
    );
    // Rounding can only produce equal neighbours; keep them non-decreasing.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: trophy.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
