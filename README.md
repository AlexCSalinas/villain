# Villain

Reads hand histories, works out what kind of player each opponent is, prices
what their leaks are worth, and remembers them the next time they sit down.

Built for the sample sizes real games actually give you — a couple hundred
hands, not a couple hundred thousand.

## Install

Needs Python 3.11 or newer.

```bash
git clone https://github.com/aroraarnav/villain.git
cd villain

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

That installs two commands, `villain` and `villain-ui`. To check it works:

```bash
pytest                             # 118 tests
```

## Use it

### The interface

```bash
villain ui                         # opens http://127.0.0.1:8766
```

Two tabs.

**Session** — drag a hand history onto the drop zone. You get the full read
straight away: what kind of player each opponent is, what to do about them, and
a skill rating. **Nothing is saved.** Close the tab and your database is exactly
as it was. Use this to look at a game you just played without committing to
anything.

**Database** — everyone you have ever imported, accumulating across sessions.
Click a player for their profile. A player you have 800 hands on reads far more
sharply than one you have 80 on, which is the entire point of keeping the
database.

To move a session into the database, hit **Add to database** on the session tab.
That is the only moment the tool asks you anything — see *Saving a session*
below.

Hover the **ⓘ** next to anything you do not recognise. Every statistic explains
what it counts and what a high *and* a low value mean for you; every piece of
vocabulary (`thin`, `usable`, `tentative`, `bb/100`, `breakeven`) explains
itself.

### The command line

```bash
villain scout ~/Downloads/poker-now-hands-*.json   # read a file, save nothing
villain import ~/Downloads/poker-now-*.json        # read it and store it
villain players                                    # who is in the database
villain profile DavidMazour                        # the full read
villain profile DavidMazour -v                     # plus deviations and timing
villain profile DavidMazour --json                 # machine-readable
villain link --suggest                             # find accounts that may be one person
villain link 4 7                                   # merge player 7 into player 4
villain note DavidMazour "tilts after losing a big pot"
villain fit                                        # learn from your own database
villain rebuild                                    # recompute everything from stored hands
```

Every command takes `--db PATH`; the default is `~/.villain/villain.db`.

### Where to get hand histories

**PokerNow** (currently the only supported format): open the game log and use
the export button; you get a `poker-now-hands-game-*.json` file. Both the
website and the CLI take it as-is.

Other sites need a parser in `villain/parsers/`. The registry sniffs formats by
file, so adding one touches nothing downstream.

### Saving a session

Identity is where a profiler quietly destroys its own data — merge two people
and both profiles become fiction — so saving is the one moment the tool asks
questions instead of guessing:

* **"Is *Dave (new laptop)* the same player as *DavidMazour2*?"** Same account
  id, new display name. Defaults to **yes**: one id almost always means one
  person who renamed themselves.
* **"Are *Arnav* and *Arnav2* the same person?"** Two different account ids that
  look alike. Defaults to **no** — merging two real players is the more
  expensive mistake.

Players dealt into the same hand are never offered as a merge, whatever their
names look like.

### Reading the output

The number to act on is **bb/100** next to each exploit: roughly what that leak
is worth to you per 100 hands if you attack it every time. Sorted by value, so
the top line is where the money is.

Each read is labelled `tentative`, `likely` or `strong`. Those are not decoration
— they say how much of the read comes from their actual hands rather than from
assumptions about players in general. A `tentative` read on 12 spots is worth
knowing and not worth rebuilding your game around.

If a player shows no leaks, that is usually "not enough hands yet" rather than
"unexploitable". The tool says so rather than inventing something.

## The problem this is actually solving

A home game session is about 200 hands. That is 20 to 40 observations of any
postflop statistic. A tracker will happily tell you the villain folds to 100%
of turn bets because they folded the only three they faced, and betting every
turn on that basis is how you donate to a player who is perfectly normal.

So the hard part is not computing statistics. It is knowing which of them mean
anything yet. Three decisions follow from that:

**Every number is shrunk toward a population prior and carries its own
uncertainty.** A frequency is a Beta posterior, not a fraction. Three
observations barely move it; three hundred make the prior irrelevant. Nothing
is ever reported as a bare percentage.

**Table size is part of a player's identity, not a footnote.** 55% VPIP is a
nit heads-up, a normal three-handed player and a maniac at a full ring.
Statistics are bucketed by table size when they are *collected*, so the same
person's heads-up and three-handed play never pool into an average describing
neither. Where a player is thin in one format, their play in the others is used
as a personal prior -- discounted, because those are related games, not the
same one.

**Reads have to be earned by data.** Some population frequencies already sit
near the point where an exploit breaks even, so a rule that fires on the
estimate alone would flag players nobody has ever observed. Every leak reports
how far the evidence moved it away from the prior, and reads are graded
`strong` / `likely` / `tentative` rather than silently dropped.

## What comes out

Real output, from a 215-hand PokerNow session:

```
------------------------------------------------------------------------------
Arnav2  --  heads-up, 183 hands (usable)
------------------------------------------------------------------------------
READ: TAG  (confidence 51%)
  Solid and hard to exploit; frequencies sit close to the field.

  No large leak to attack. Play position, take the small edges, and look for
  the money at another seat.
  (also plausibly: trapper 20%, station 15%)

EXPLOITS  (1 found)
  [tentative] Folds too often to river bets  ~0.1 bb/100
      51% vs 40% breakeven  (field 45%, n=16)
      Bluff every river you reach with a busted hand, and size up -- they are
      folding to the decision rather than to the price.

SKILL: strong (69/100)   confidence 54%
  Solid and hard to attack, with only narrow leaks to work on.
    showdown judgement         ##############....  77.3
    hand selection             ###############...  81.5
    discipline vs bets         ###############...  82.8
    bet sizing                 ###############...  85.3
    postflop aggression        #################.  94.9
    preflop aggression         ##################  97.6  raises 73% of hands played
    resistance to exploitation ##################  97.7  ~0.1 bb/100 available
  observed -1.0 bb/100, shrunk and all-in adjusted -0.2 bb/100
```

Note what it does *not* claim. 183 heads-up hands support a bucket with 51%
confidence and one tentative leak worth a tenth of a big blind per 100 -- that
is what a session of this size actually contains, and a tool reporting more
would be making it up. The numbers sharpen as sessions accumulate against the
same player.

Plus `--json` for anything that wants to consume it programmatically.

## Buckets, and why they are defined the way they are

Eight archetypes -- `nit`, `station`, `overfolder`, `maniac`, `lag`, `tag`,
`limper`, `trapper`. Each is a *plan*, not a personality: "station" is the
bucket whose plan is "value bet thin and stop bluffing".

Prototypes are stored as deviations from the population **in log-odds**, which
sounds fussy and is the difference between working and not. Frequencies are
bounded, so adding percentage points to a 70% base is not the same size of
change as adding them to a 24% base. In linear space the same prototype
produces a six-handed nit who plays 2% of hands; in log-odds it produces 9.5%
at six-handed and 44% heads-up, which is what "nit" meant in both cases.

Matching is a likelihood, not a distance. Each archetype implies a frequency
for every feature, and the raw counts are scored against it with an
overdispersed Beta-Binomial. This matters because the obvious approach --
shrink the stats, then measure distance to each prototype -- counts the
uncertainty twice, and every thin sample collapses onto whichever prototype
sits in the middle. Every archetype is scored over the same feature set, since
a prototype scored only on the features it happens to mention wins by
mentioning fewer.

The output is a posterior over all eight. Players genuinely sit between
buckets, and a forced label invites a plan the evidence cannot carry.

## Leaks are priced from pot odds, not from the field

A leak fires on an **absolute** threshold, because what makes a tendency
exploitable is arithmetic, not fashion:

* a bluff of size `f` breaks even at a fold frequency of `f / (1 + f)` -- 33%
  at half pot, 40% at two thirds, 50% at pot;
* below a third folds, even the cheapest bluff worth making loses money, so
  the exploit inverts from pressure to thin value;
* a steal risking `r` to win `p` breaks even at `r / (r + p)`.

Folding 55% to a two-thirds pot bet is exploitable whether or not everyone else
in the pool folds 55% too. Population comparisons are for *identifying* a
player, and appear in the report as context only.

Severity is the estimated big blinds per 100 hands from taking the exploit,
computed from that player's own average pot sizes and how often the spot comes
up. `CAPTURE` in `exploits.py` scales it to the share of spots you can
realistically convert -- you cannot bluff a river you reached with the nuts.
That constant is an assumption, stated in the source rather than buried, and
severities are best read as a ranking of what to attack first.

## Skill

Rating a player by results is rating their luck. Results carry the smallest
weight here, and only after all-in pots are rescored by equity so a cooler and
a punt stop looking alike. The rating is built from:

* **fundamentals** -- distance from competent play for that table size, with
  asymmetric penalties where the errors are asymmetric (playing too tight
  leaves value behind; playing too loose bleeds money);
* **resistance to exploitation** -- the total bb/100 the exploit layer can find
  against them, weighted by sample size, because "no leaks found" and "no leaks
  yet findable" are the same number and only one is a compliment.

Every component reports its own score and weight, so a rating reads as a
diagnosis rather than a verdict. Low confidence means "we do not know yet", and
the score is pulled toward the middle to say so.

## The interface

```
villain ui            # http://127.0.0.1:8766
```

Standard library HTTP server, one self-contained module, no new dependencies.
Two tabs, because there are two questions.

**Session** answers "who am I playing right now". Drop a file on it and you get
the full read — archetypes, priced leaks, ratings — with **nothing written
anywhere**. A session you never save leaves no trace on the database.

**Database** answers "who is this, and what do I know about them" across every
session ever imported.

Saving a session is a separate, optional step, and it is the only moment the
tool asks anything. Identity is where a profiler quietly destroys its own data:
merge two people and both profiles become fiction; split one person and half of
what you know is thrown away. So the save flow surfaces every identity call it
is unsure about instead of guessing:

* **One account id, a new display name.** The PokerNow case — same id, now
  called something else. Almost always one person renaming themselves, so it
  defaults to *same player*. Answer no and they are kept apart from then on,
  with the account keyed as `<account>#<name>` so neither loses the hands
  already attributed to them.
* **Two account ids that look like one person.** `Arnav` and `Arnav2`. This
  defaults to *different people* — merging two real players is the more
  expensive mistake and the evidence for it is weaker.

Accounts dealt into the same hand are never offered as a merge, in either
direction.

Every profile is one player at a time, chosen from a tab strip, rather than a
page you scroll. Within a profile the default view is short — the read, what to
do about it, the score, and six headline numbers. Everything else (the full
statistic list, the skill breakdown, how the archetype evidence splits) sits
behind a disclosure, because a mid-session read that requires scrolling is a
read you will not use.

Anything the tool says in shorthand carries an **ⓘ** on hover: what the
statistic counts, and what a *high* and a *low* value each mean for you. Both
directions are there deliberately — for most of these, high and low are both
exploitable and they call for opposite play, and getting that backwards costs
more than not knowing the number at all. The same applies to the vocabulary:
`thin`, `usable`, `tentative`, `bb/100`, `breakeven` and `available` all
explain themselves on hover. Definitions live in `glossary.py` as data, and a
test fails if any statistic reaches the screen without one.

**Reset** lives on the database tab and asks you to type `delete everything`.
It removes hands, players, and every merge and rename decision. Export files
are untouched, so the statistics can be rebuilt — the identity decisions cannot.

### Charts

Every read in this tool is the same shape — a frequency, an uncertainty, and a
threshold that decides whether it matters — so the profile view uses one
repeated mark: a wash for the 95% credible range, a dot for the estimate, a
hairline tick for the field, and a warm tick for breakeven. When the dot sits
past the warm tick, there is money there.

The palette is monochrome, stepped by confidence, and both ramps were run
through a contrast and colour-blindness validator rather than eyeballed. Two
results worth recording: encoding exploit *kind* (pressure versus value) as a
second hue failed CVD separation at these lightnesses, so kind is carried by a
label instead; and the lightest step of the confidence ramp had to be darkened
to clear the 2:1 floor against the panel surface.

## Remembering people

Hands are the source of truth; statistics are a disposable cache. Stat
definitions change, and `villain rebuild` recomputes every profile from stored
hands rather than leaving old players wrong until they happen to sit down
again.

Identity is separate from account. The same human is `DavidMazour` at one table
and `DavidMazour2` at the next, so site accounts are aliases pointing at an
internal player. Candidate merges come from name similarity (after stripping
case, punctuation and trailing digits) and from a Bayes factor over their
statistics -- the probability both samples came from one player against the
probability they came from two, which stops two tight players being merged just
for both being tight.

One constraint overrides everything: accounts dealt into the same hand are
different people, whatever their names look like. Those pairs are recorded at
import and can never be linked. Nothing is merged automatically -- a wrong
merge corrupts two profiles at once and costs far more than a missed one.

## Learning from your own pool

`villain fit` runs three models and tells you which ones your data can support:

* **priors** re-estimated from your own players by a Beta-Binomial moments fit,
  so a home game stops being measured against an online population. The spread
  between your players sets how much a new player's sample is trusted;
* **clusters** -- a Gaussian mixture over profiles, component count chosen by
  BIC, which finds the player types actually present in your game rather than
  the ones a textbook expects. Needs 25+ profiles;
* **hand strength** -- a gradient boosting model mapping a line (street,
  action, sizing, position, board texture, time taken) to the strength of the
  hand behind it, trained on revealed cards, with per-player residuals from
  out-of-fold predictions. "Shows up 20 percentile points weaker than the field
  on the lines they take" is directly actionable in a way that a betting
  frequency is not. Needs 300+ revealed decisions.

Each refuses rather than returning something that looks authoritative and is
not.

## Known limitations

* **Showdown data is biased.** Villains' cards are revealed only at showdown,
  and hands reaching showdown skew toward calling lines and away from the
  bluffs that took the pot down uncontested. The strength model therefore
  *underestimates* how weak betting ranges are. The exporting player's own
  cards are visible on every hand and those rows are marked unbiased, but the
  bias is reduced, not removed.
* **Severity constants are assumptions.** The breakeven thresholds are derived;
  the capture fractions are judgement calls.
* **Side pots are not modelled.** All-in equity with three or more players of
  different stacks is approximate, and those hands are flagged rather than
  trusted.
* **Built-in priors are pool-agnostic.** They describe a generic online
  population. Run `villain fit` once your database has enough players.
* **PokerNow is the only parser so far.** The format registry in
  `villain/parsers/` takes new sites without touching anything downstream.

## Layout

| module | what it owns |
| --- | --- |
| `model.py` | canonical hand representation, positions, serialisation |
| `parsers/` | site formats; `pokernow.py` decodes the numeric opcode log |
| `cards.py`, `equity.py` | vectorised 7-card evaluator, all-in equity |
| `stats.py`, `features.py` | additive sufficient statistics per hand |
| `priors.py`, `profile.py` | shrinkage, per-regime and cross-regime priors |
| `archetypes.py`, `exploits.py`, `skill.py` | buckets, priced leaks, rating |
| `cluster.py`, `reads.py` | models learned from your own database |
| `db.py`, `identity.py` | persistence, aliases, merge safety |
| `report.py`, `cli.py` | terminal output and commands |

## Tests

```
pytest
```

97 tests. The parser suite checks that every hand balances to the cent, which
is what proves the opcode decoding is right; the evaluator is verified against
brute-force best-of-five on random deals; and several tests in
`test_profiling.py` are regressions named for the modelling mistakes that
produced them.
