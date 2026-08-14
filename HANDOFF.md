# Handoff

**207/207 tests pass. Everything committed and pushed to `main`.**

Live database: `~/.villain/villain.db` — 12,608 hands, 73 players, 20 sittings.

---

## The one thing to read first

The archetype prototypes have been tuned, twice, against **six players whose
labels were supplied by hand**. Those six labels have been *inside* the tuning
loop from the beginning, and that is the root cause of most of the wasted
effort in this project's recent history:

* a "TAGs are tighter than the field" trait turned out to be a two-player
  artifact and ejected a known TAG;
* a `fold_vs_bet:river` trait was set with the **wrong sign** and survived
  because it happened to fit four players;
* softening `lag` to make it reachable turned it into the new centroid magnet;
* `raise_share` was added on a sound argument and moved nothing.

`villain validate` now exists to break that loop. It splits each player's hands
into interleaved disjoint halves, builds a profile from each, and scores whether
the posterior from one half predicts what the other half supports. The target is
observable, needs no labels, and the scorer is a plain unweighted likelihood at
fixed concentration so it cannot become another tuned knob.

**Tune against the harness. Use the six labels only as a smell test afterwards.**

Current standing: `Arnav` tag ✓, `Arav` tag ✓, `Vik` lag ✓, `nuj` ✗ (reads
tag), `Grayson` ✗ (reads tag), `Arjun` ✗ (reads tag). **3 of 6, and it did not
move all session** despite several real bug fixes. Nobody yet knows why `nuj`
reads TAG.

---

## What the harness says now

```
villain validate

54 players scored on disjoint halves
  log loss                1.691
  top-1 accuracy          0.556
  calibration error       0.024
  mean stated confidence  0.532
  halves agree            0.519
```

Swept across the discount, calibration error bottoms out at exactly the value
now in use:

| discount | log loss | accuracy | calib err |
|---|---|---|---|
| 0.55 (old) | 2.766 | 0.611 | 0.158 |
| 0.20 (now) | 1.691 | 0.556 | **0.024** |
| 0.15 | 1.548 | 0.500 | 0.038 |

Note accuracy **does** fall, 0.611 → 0.556. The trade is 5.5 points of top-1
accuracy for 6.6x better calibration. Defensible for a tool whose identity is
honesty about uncertainty, but it is a trade.

---

## Fixed recently (do not re-investigate)

Verified against the code, not against notes:

* **Severity was 100x too small** — `gain x spots_per_100` is already bb/100.
* **`link()` corrupted `distinct_pairs`**, letting same-hand accounts merge.
  Existing databases self-repair on open.
* **Evidence counted a 60-hand window**, so `count` was always the cap and
  `hits` was computed inside it: 4 limps in 6,210 hands showed as "0 of 60".
* **Fitted priors were never refreshed** when definitions changed. `raise_share`
  had 0 fitted rows while carrying importance 2.6 — refitting changed **26 of
  68** labels.
* **Derived features had no fitted prior at all.** `aggression:*` (combined
  importance 4.0) was measured against a built-in 0.280 where the pool is 0.208.
  That also un-broke `barrels_flop`, which fired on **0 of 54** players.
* **`match()` scored borrowed cross-regime pseudo-counts as observations** —
  24% of the counts, and the most-wrong players borrowed most. Estimates now
  carry `native_opps`.
* **`CORRELATION_DISCOUNT` 0.55 → 0.20**, derived two ways. Nine players sat at
  1.00 confidence; none do now.
* Bomb pots no longer manufacture preflop opportunities; ante and raked hands
  are no longer silently dropped; CSRF on destructive endpoints; side-pot EV
  capped by eligibility; check-raise double-counting; delayed c-bets.
* **`.act` class collision** — `button.act` never declared `display`, so *every
  button in the app* was laid out as a 4-column grid with a 44px first track.

---

## What is left

### Accuracy — all three need the harness as the objective

1. **Wire `Profile.fold_accuracy()` into `match()`.** The measure is built and
   separates the known labels cleanly (strong ≤0.064, weak ≥0.069) where no
   signed frequency does — every other trait is a *signed* deviation, so
   folding far too much and far too little land at opposite ends while both are
   the same mistake. It is **not** wired in: at a defensible weight it is worth
   ~0.5 nats against margins many times that, and turned up far enough to move
   the wrong players it stops the prototypes recovering their own frequencies.
   Score it through `villain validate`, not against the six labels.
   *Note: it must use a fixed reference (`CORRECT_FOLD = 0.44`), not each
   player's own faced sizes — the personalised version inverts the signal,
   because a player who calls too much gets shown smaller bets, which lowers
   his own breakeven until he clears it.*
2. **Retune `maniac` vs `lag`** — 1.9 nats apart; maniac is "lag turned up",
   differing in degree not kind. Proposal: give maniac negative `wwsf`/`wsd`,
   so the distinction becomes whether the aggression *works*.
3. **`fold_vs_bet:river` carries importance 3.4** — the highest in the table —
   on a median of 25 opportunities, and it decides 89% of one player's entire
   margin. Tuned on four players.

Also open: **10 exploit thresholds fire on nobody**, nine of them unreachable by
construction (`bluffs_rivers` asks 0.40 where the pool best is 0.23; the whole
timing family sits above its own ceiling). The docstring is now honest about
which thresholds are derived and which are chosen.

### The nuj problem

`nuj` reads TAG and should not. What is known:

* His fold accuracy is 0.097 against strong players' ≤0.064 — the measure sees
  it, the matcher does not.
* Priced at the size you would actually bet (two-thirds, not the half-pot he is
  shown), he under-folds flops by **3.3pp** — real, but inside the 5pp `MARGIN`
  that stops the tool inventing leaks.
* His own description of the leak is **compound**: over-calls flops *and
  therefore* arrives on later streets with a range far wider than his line
  claims. The tool prices each street independently. `playbook.COMBINATIONS`
  already exists for leaks that reinforce each other and is the right home for
  this — it is not a new rule.

### Design — small

Dead CSS (~25 lines never emitted: `.read-grid`, `.skill-side .skill-head`,
`.legend`/`.swatch`, `.cards`); the replay sheet reuses the evidence headline as
its own title; modal header should be sticky over a long hand list;
`th.sorted::after` shows the same caret for both sort directions.

### Net new — nothing built yet

Ranked by value per effort:

1. **`villain backtest`** — walk-forward scoring of the tool's own leak
   predictions. A prototype run over the real data gave **strong 86% / likely
   76% / tentative 57%**, monotone and well separated. That turns the project's
   central claim from an argument into a scoreboard and needs no new modelling.
   Caveat that must ship with it: if you acted on a flagged leak the opponent
   may have adjusted, so a read that stops holding can look like a bad
   prediction rather than a successful exploit.
2. **Fix the hand-strength model's per-player reads.** They have never printed
   for anyone, on any database: `_cmd_fit` passes `store.stored_hands()`, which
   deliberately preserves **raw site account ids** (a test enforces this — the
   hands are the source of truth and a merge must not rewrite them), while the
   lookup asks for internal player ids. **Fix at the call site**, following
   `player_hands`. Also `reads.FEATURES` omits the `unbiased` flag, so the
   baseline blends showdown-selected villain rows with non-selected hero rows.
   Residuals must be computed **per line**, not pooled — pooled, only 2 of 30
   players clear the reporting bar; split by street x action, 19 cells across 11
   players do.
3. "What would it take to confirm this" — spots needed to cross the bar, or
   never. Pure arithmetic on the posterior already computed.
4. Hero fold-strength (cards known in 98.8% of 6,433 hands — but note the
   preflop range audit was prototyped and correctly finds **nothing**: opening
   ranges are monotone, 0 violations in 289 comparable pairs).
5. Binary high/low board texture on the fold rules (31 players qualify; the
   four-way split does not — only 2 do).
6. `villain table <names>` lineup briefing — but **never** as an expected win
   rate: per-session bb/100 has SD 183, so separating two lineups needs a
   ~229 bb/100 gap before it is detectable.

---

## Things that will bite you

* **`spread_of()` ignores its `table_regime` argument.** Fitting it from the
  `(mean, strength)` pair was tried and reverted: where `fit_empirical` cannot
  separate the pool it returns a large strength, implying a *tiny* spread, and a
  tiny spread **amplifies** every deviation measured against it. Ten features hit
  that and drove confidence back to 1.00 on their own.
* **Priors apply when a profile is read, not when books are built.** So
  refitting takes effect immediately and a rebuild afterwards is pure waste —
  one was removed for costing ~30s of blocked window per import.
* **A validation harness needs validating.** The first version of
  `villain validate` split the *counts* rather than the hands, giving two halves
  with identical rates that agreed 100% by construction.
* Your winrates are implausible for real poker (+77 bb/100 over 6,210 hands).
  Treat the ordering as signal and the magnitude as suspect.

## Re-verify

```bash
cd /Users/arnavarora/Projects/villain && source .venv/bin/activate
pytest -q                 # 207
villain validate          # calibration on unseen hands
villain ui                # http://127.0.0.1:8766
```
