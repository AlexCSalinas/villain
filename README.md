# Villain

Reads hand histories, works out what kind of player each opponent is, prices
what their leaks are worth, and remembers them the next time they sit down.

## Why this is hard

A home game session is about 200 hands — 20 to 40 observations of any postflop
statistic. A tracker will tell you the villain folds to 100% of turn bets
because he folded the only three he faced, and betting every turn on that basis
is how you donate to a normal player. The hard part is not computing statistics
but knowing which of them mean anything yet, and everything below follows from
that: this is built for a couple of hundred hands, not a couple of hundred
thousand.

## Install

Needs Python 3.11 or newer.

```bash
git clone https://github.com/aroraarnav/villain.git
cd villain

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

pytest                             # 167 tests
```

That installs two commands, `villain` and `villain-ui`. The parser suite checks
that every hand balances to the cent, which is what proves the opcode decoding
is right; several tests in `test_profiling.py` are regressions named for the
modelling mistakes that produced them.

## The interface

```bash
villain ui                         # http://127.0.0.1:8766
```

Standard library HTTP server, one self-contained module. Three tabs, because
there are three questions.

* **Session** — who am I playing right now. Drop a hand history on the page and
  the read comes back at once. **Nothing is written anywhere.**
* **Database** — who is this, across every session ever imported. One profile
  per player, and 800 hands read far more sharply than 80.
* **Leaderboard** — every player you have recorded, ranked by skill score, with
  the bb/100 you can attack them for alongside.

A profile shows one player at a time from a tab strip: the read, what to do
about it, the score, six headline numbers, and the rest behind a disclosure,
because a mid-session read that requires scrolling is one you will not use.
Every shorthand carries an **ⓘ** giving what the statistic counts and what
*high* and *low* each mean — both, since both are usually exploitable and call
for opposite play. Definitions live in `glossary.py` as data, and a test fails
if a statistic reaches the screen without one. **Reset** makes you type `delete
everything`; exports are untouched, so statistics can be rebuilt, but the merge
and rename decisions cannot.

### Saving a session

**Add to database** is the only moment the tool asks you anything. Identity is
where a profiler destroys its own data — merge two people and both profiles
become fiction, split one and half of what you know is gone — so it asks about
every call it is unsure of rather than guessing.

* **One account id, a new display name.** The PokerNow case, almost always a
  rename, so it defaults to *same player*. Answer no and they stay apart, keyed
  `<account>#<name>` so neither loses its existing hands.
* **Two account ids that look like one person** — `Arnav`, `Arnav2` — default
  to *different people*: the merge is the more expensive mistake on the weaker
  evidence.

Accounts dealt into the same hand are never offered as a merge, whatever their
names look like.

## The command line

Every command takes `--db PATH`; the default is `~/.villain/villain.db`.

| command | what it does |
| --- | --- |
| `villain scout FILE...` | read a file, save nothing (`--min-hands N`, default 20; `-v`) |
| `villain import FILE...` | read it and store it (`--quiet`) |
| `villain players` | who is in the database (`--min-hands N`) |
| `villain profile NAME` | the full read (`-v` for deviations and timing, `--json`, `--narrate`) |
| `villain profile NAME --by-table` | split by table size instead of pooling (`--regime hu\|3max\|6max\|full`) |
| `villain link --suggest` | find accounts that may be one person |
| `villain link KEEP ABSORB` | merge player `ABSORB` into player `KEEP` |
| `villain note NAME "tilts after a big pot"` | attach a note to a player |
| `villain fit` | learn priors, clusters and hand strength from your own database (`--min-players N`, default 8) |
| `villain rebuild` | recompute every profile from stored hands |
| `villain ui` | serve the web interface (`--port N`, default 8766; `--no-browser`) |

**PokerNow** is currently the only supported format: open the game log and use
the export button, which gives a `poker-now-hands-game-*.json` file that both
the website and the CLI take as-is. Other sites need a parser in
`villain/parsers/`; the registry sniffs formats by file content, so adding one
touches nothing downstream.

### Optional: a plain-English summary

Everything else is deterministic: the same hands give the same read, and no
figure on screen came from anywhere but the arithmetic. One optional extra runs
a language model over the finished profile and writes a short briefing joining
the findings together — off unless configured, and **local** by default, which
keeps it free, offline, and keeps opponent profiles on your own machine. The
same narrator sits behind the **generate detailed description** button on a
player's profile in the web interface.

Settings come from the environment, falling back to **`~/.villain/env`** — a
plain `NAME=value` file that lives outside the project directory on purpose. A
key that never sits under the working tree cannot be committed by an
absent-minded `git add -A`. Make it readable by you alone:

```bash
mkdir -p ~/.villain && chmod 600 ~/.villain/env
```

| variable | meaning |
| --- | --- |
| `VILLAIN_LLM_MODEL` | model name (default `llama3.2`) |
| `VILLAIN_LLM_URL` | any OpenAI-compatible `/chat/completions` endpoint (default Ollama on localhost) |
| `VILLAIN_LLM_KEY` | bearer token, if the endpoint needs one |

Local, with nothing leaving the machine:

```bash
brew install ollama && ollama serve
ollama pull llama3.2
VILLAIN_LLM_MODEL=llama3.2 villain profile DavidMazour --narrate
```

Or a hosted free tier — faster and no install, at the cost of sending opponent
profiles to somebody else. Gemini, in `~/.villain/env`:

```
VILLAIN_LLM_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
VILLAIN_LLM_MODEL=gemini-flash-latest
VILLAIN_LLM_KEY=your-key-here
```

Use a floating alias like `gemini-flash-latest` rather than a pinned version:
pinned Gemini models retire and start returning 404 to a tool that was working
last month.

The model gets a fact sheet built from the computed profile and may not add
figures of its own: any number not in the facts discards the whole response in
favour of the written text. Rounding 51% to "about half" is fine, deciding they
fold 70% is not, and the prose does not tell you which — so the check is
mechanical rather than a matter of trust.

## Reading the output

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
    showdown judgement         ##############....  77.3
    hand selection             ###############...  81.5
    bet sizing                 ###############...  85.3
    preflop aggression         ##################  97.6  raises 73% of hands played
    resistance to exploitation ##################  97.7  ~0.1 bb/100 available
  observed -1.0 bb/100, shrunk and all-in adjusted -0.2 bb/100
```

Note what it does *not* claim: 183 hands buy a bucket at 51% confidence and one
tentative leak worth a tenth of a big blind per 100, which is what a session
this size contains.

Each exploit answers four questions — what they are doing (as behaviour, not as
a statistic), why it is exploitable (the breakeven arithmetic), what to do, and
the counter-mistake, which matters most because nearly every way of losing
money to a correct read is an over-adjustment. Leaks that compound are called
out together: folding flops too often *and* never check-raising removes both
the reason to fear betting and the cost of being wrong.

Leaks are sorted by **bb/100**, what one is worth per 100 hands if you attack
it every time, and labelled `tentative`, `likely` or `strong` by how much comes
from this player's hands rather than from the prior. No leaks usually means not
enough hands yet, and the tool says so rather than inventing something.

### Checking a read

Every exploit carries a **see the N hands** button — board, what they did, what
it cost, whether it counted — and any of them replays street by street. Nothing
extra is stored: contributing hands are found by replaying each hand through
the same extraction the statistics use, so the evidence cannot drift from the
number, being the same code. It is also the fastest way to catch the tool being
wrong; building it surfaced VPIP counting once per preflop *decision* instead
of once per hand.

## How it works

**Everything is shrunk toward a population prior and carries its own
uncertainty.** A frequency is a Beta posterior, not a fraction: three
observations barely move it, three hundred make the prior irrelevant.

**Table size is part of a player's identity.** 55% VPIP is a nit heads-up, a
normal three-handed player and a maniac at a full ring, so statistics are
bucketed by table size as they are *collected*. Reporting them that way is
unreadable, so each player still gets **one** profile, pooled in log-odds: each
table's counts become a deviation from that table's own population, translated
onto the scale of the table they play most, then discounted, because related
games are not the same game. `--by-table` shows the split.

**Reads are earned by data.** Some population frequencies already sit near an
exploit's breakeven point, so a rule firing on the estimate alone would flag
players nobody has observed. Every leak reports how far the evidence moved it
from the prior, and reads are graded rather than dropped.

### Buckets

Eight archetypes — `nit`, `station`, `overfolder`, `maniac`, `lag`, `tag`,
`limper`, `trapper` — each a *plan* rather than a personality: "station" is the
bucket whose plan is "value bet thin and stop bluffing".

Prototypes are deviations from the population **in log-odds**, which is the
difference between working and not: frequencies are bounded, so points on a 70%
base are not the same change as points on 24%. In linear space the same "nit"
prototype produced a six-handed player who plays 2% of hands; in log-odds it
lands at 44% heads-up and 9.5% full ring, which is what "nit" meant in both.

Matching is a likelihood, not a distance: raw counts are scored against each
archetype's implied frequencies with an overdispersed Beta-Binomial. Shrinking
first and then measuring distance counts the uncertainty twice, and thin
samples collapse onto the prototype in the middle. All eight are scored over
the *same* features — one scored only on the features it mentions wins by
mentioning fewer — and the result is a posterior, because players sit between
buckets.

### Leaks are priced from pot odds, not from the field

A leak fires on an **absolute** threshold, because what makes a tendency
exploitable is arithmetic, not fashion:

* a bluff of size `f` breaks even at a fold frequency of `f / (1 + f)` — 33% at
  half pot, 40% at two thirds, 50% at pot;
* below a third folds, even the cheapest bluff worth making loses money, so the
  exploit inverts from pressure to thin value;
* a steal risking `r` to win `p` breaks even at `r / (r + p)`.

Folding 55% to a two-thirds pot bet is exploitable whether or not the rest of
the pool folds 55% too; population comparisons only *identify* a player, and
appear as context. Severity is the estimated bb/100 from taking the exploit,
from that player's own pot sizes and how often the spot comes up, scaled by
`CAPTURE` in `exploits.py` to the share of spots you can realistically convert
— you cannot bluff a river you reached with the nuts. That constant is an
assumption stated in the source rather than buried, so severities are best read
as a ranking of what to attack first.

### Skill

Rating a player by results is rating their luck, so results carry the smallest
weight, and only after all-in pots are rescored by equity so a cooler and a
punt stop looking alike. The rest is **fundamentals** — distance from competent
play for that table size, penalised asymmetrically where the errors are — and
**resistance to exploitation**, the bb/100 the exploit layer can find, weighted
by sample size, because "no leaks found" and "no leaks yet findable" are the
same number and only one is a compliment.

### Remembering people

Hands are the source of truth and statistics are a disposable cache:
definitions change, and `villain rebuild` recomputes every profile from stored
hands rather than leaving old players wrong until they sit down again.

The same human is `DavidMazour` at one table and `DavidMazour2` at the next, so
accounts are aliases pointing at an internal player. Candidate merges come from
name similarity (case, punctuation and trailing digits stripped) and a Bayes
factor over their statistics — one player against two — which stops two tight
players being merged for nothing more than both being tight. Nothing merges
automatically, and accounts dealt into the same hand can never be linked.

### Learning from your own pool

`villain fit` runs three models and says which ones your data can support:
**priors** re-estimated from your own players by a Beta-Binomial moments fit,
so a home game stops being measured against an online population and the spread
between your players sets how much a new sample is trusted; **clusters**, a
Gaussian mixture over profiles with component count by BIC, needing 25+
profiles; and **hand strength**, gradient boosting from a line (street, action,
sizing, position, board texture, time taken) to the strength behind it, trained
on revealed cards, needing 300+ revealed decisions. Each refuses rather than
returning something authoritative-looking and wrong.

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
| `model.py`, `parsers/` | canonical hands; `pokernow.py` decodes the opcode log |
| `cards.py`, `equity.py` | vectorised 7-card evaluator, all-in equity |
| `stats.py`, `features.py` | additive sufficient statistics per hand |
| `priors.py`, `profile.py` | shrinkage, per-regime and cross-regime priors |
| `archetypes.py`, `exploits.py`, `skill.py` | buckets, priced leaks, rating |
| `playbook.py`, `narrate.py` | written advice, optional LLM summary |
| `evidence.py`, `replay.py` | the hands behind a number |
| `cluster.py`, `reads.py` | models learned from your own database |
| `db.py`, `identity.py` | persistence, aliases, merge safety |
| `analyze.py`, `glossary.py`, `report.py` | the payload the CLI and UI both render |
| `cli.py`, `web.py` | commands, local web UI |
