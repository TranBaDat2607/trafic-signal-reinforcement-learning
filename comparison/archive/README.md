# Archived comparison results — superseded

These two directories hold the results published before the metric definitions
were unified. **They are not comparable to anything produced by the current
code, and `compare_results.py` will refuse to load them** (they carry no
`metrics_version`). They are kept only as the provenance of the two PNGs that
were once the repository's headline results.

Superseded by the fix for PROBLEMS.md P0-1 / P0-2 / P0-3.

## Why the numbers cannot be used

Both directories were generated when the RL path and the baseline path each had
their own aggregation:

- **`cumulative_rewards`** — the baseline summed *every* per-step reward. Per-step
  reward is a telescoping difference `W(t-1) - W(t)`, so the sum collapses to
  `-W_final`, which is `0.0` in every episode here. The RL side summed only the
  negative rewards. The two were plotted on one axis; neither measured
  controller quality.
- **`cumulative_waits`** — the baseline summed an *already accumulated* waiting
  time once per step, which double-counts. Episode 0 of `comparison_du10`
  records `316843.0`; the true delay for that episode is
  `avg_queue 3.369 × 5400 steps = 18,195` vehicle-seconds, about 17× smaller.
  The RL side's series under the same key was genuinely vehicle-seconds, just
  mislabelled `"Cumulative delay (s)"`.
- **`avg_queues`** — the one series that was measured identically on both sides,
  and therefore the only trustworthy number in either directory.

## Additionally

- **`comparison_du100/` used a broken signal plan.** All eight phases in the net
  file carried `duration="100"`, including the four yellow phases. A 100-second
  yellow light is not a traffic-engineering choice. Its `RL +90.2%` headline was
  measured against that. See PROBLEMS.md P0-4.
- **The RL half was never committed** — only the baseline `.json`/`.txt` files
  are here. The PNGs are the sole record of the agent's numbers. See PROBLEMS.md
  P0-6.
- **Both PNGs come from the same 10-episode run** with `epsilon = 1 - episode/10`,
  which reaches 0 only at the final episode. There was no exploitation phase and
  therefore no trained policy to report on. See PROBLEMS.md P0-5.

Regenerating a defensible comparison requires P0-4 (a realistic fixed-time plan)
and P0-5 (a real training run) first.
