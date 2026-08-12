"""What each leak means, in words a player can act on.

:mod:`villain.exploits` decides *whether* a tendency is exploitable. This module
says what to do about it. They are kept apart because they change for different
reasons: the arithmetic moves when the maths is wrong, the language moves when a
human reads it and is confused.

Every entry answers four questions, in the order a player needs them:

``behaviour``
    What this person is actually doing at the table, described as behaviour
    rather than as a statistic. "They fold the turn whenever they miss" is
    usable at the table; "fold_vs_bet:turn 61%" is not.
``why``
    The mechanism. Why does this cost them money and hand it to you? Tied to
    the breakeven arithmetic wherever possible, because that is where the edge
    actually comes from and a player who understands it can adapt when the
    number shifts.
``do``
    Concrete actions, with streets and sizes. Something executable on the next
    hand, not a principle to bear in mind.
``dont``
    The counter-mistake. This is the field most tools omit and the one that
    saves the most money: nearly every way of losing money to a correct read
    is an over-adjustment. Knowing somebody folds too much does not mean
    betting every chip you have.

The tone is deliberately flat. This gets read between hands by somebody who is
also playing poker, so no throat-clearing and no hedging that does not carry
information.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    behaviour: str
    why: str
    do: str
    dont: str


#: Keyed by ``Rule.id`` in :mod:`villain.exploits`.
PLAYBOOK: dict[str, Entry] = {
    # -- preflop ------------------------------------------------------------
    "folds_blinds": Entry(
        behaviour="They give up their blind rather than play out of position. "
                  "When you raise from late position, they fold and wait for a "
                  "better spot that mostly never comes.",
        why="A raise risks a small amount to win the blinds. Once they fold "
            "more often than that price needs, the raise profits before a "
            "single card is dealt -- your cards stop mattering, because most "
            "of the time the hand ends immediately.",
        do="Raise every button and small blind against them, regardless of "
           "what you hold. Keep the size small, around 2.2 to 2.5 big blinds: "
           "you are buying folds, and a bigger raise pays more when they do "
           "wake up with something.",
        dont="Do not keep firing after they defend. A player who folds this "
             "often is only continuing with real hands, so their call means "
             "far more than a normal player's would. Steal relentlessly, then "
             "give up cheaply when they push back.",
    ),
    "folds_to_three_bet": Entry(
        behaviour="They open plenty of pots but abandon them the moment "
                  "somebody re-raises. Their opening range is much wider than "
                  "the range they will actually play a big pot with.",
        why="They are raising a range they cannot defend. Once they fold too "
            "often to a re-raise, your re-raise wins the pot outright often "
            "enough to profit with any two cards -- you are attacking the gap "
            "between what they open and what they continue with.",
        do="Re-raise their opens light, especially in position. Hands that "
           "flop well but cannot call a raise -- suited connectors, suited "
           "aces, small pairs -- are ideal, since you win immediately most "
           "of the time and have a playable hand when you do not.",
        dont="Do not re-raise into the same player twice in quick succession "
             "with nothing. Even weak players notice a pattern, and the "
             "adjustment is to start playing back at you. Space it out, and "
             "fold to their four-bet -- at this fold frequency, a four-bet is "
             "always real.",
    ),
    "no_three_bet": Entry(
        behaviour="They almost never re-raise before the flop. They call with "
                  "good hands rather than raising them, so their calls contain "
                  "everything and their raises contain almost nothing.",
        why="Two edges, in opposite directions. Their flat calls are capped, "
            "so you can open far wider and continue after the flop knowing "
            "they rarely have a monster. And when they do re-raise, it is the "
            "very top of their range -- information most players have to pay "
            "to get.",
        do="Open wider from every seat and bet the flop freely against their "
           "calls. When they finally re-raise, fold anything that is not "
           "premium; you are getting a free look at the top of their range.",
        dont="Do not confuse their passivity with weakness after the flop. A "
             "player who only calls preflop still has strong hands in that "
             "calling range -- they just did not raise them. Keep betting, "
             "but believe their flop and turn raises.",
    ),
    "limps": Entry(
        behaviour="They enter pots by calling the big blind instead of "
                  "raising. Their limping range is wide, weak, and full of "
                  "hands they are hoping to see a cheap flop with.",
        why="A limping range is capped -- strong hands would usually raise -- "
            "and it is built for seeing flops, not for playing big pots. So "
            "you profit twice: you get to raise a wide range for value "
            "preflop, and you get to bet a flop they will usually have missed.",
        do="Raise over their limps with a wide range, sized to around 4 big "
           "blinds plus one for each limper. Then bet the flop whether or not "
           "you connect -- they fold everything they miss, and they miss most "
           "of the time.",
        dont="Do not automatically continue on the turn when they call the "
             "flop. A limp-caller who calls a flop bet usually has something "
             "real. Take the free pots and stop when they show interest.",
    ),
    "no_defend": Entry(
        behaviour="They surrender their big blind far too often, folding "
                  "hands that are well worth a call against a late-position "
                  "raise.",
        why="The big blind is getting a better price than any other seat -- "
            "money is already in the pot on their behalf. Folding too much "
            "there is a guaranteed loss on every orbit, and every one of those "
            "folds is yours to collect.",
        do="Attack their big blind with any two cards from late position. A "
           "minimum raise is enough; you do not need a big size to win a pot "
           "nobody is defending.",
        dont="Do not extend this to the times they do defend. Their defending "
             "range is tighter than normal precisely because they fold so "
             "much, so a flop continuation bet has less equity against it "
             "than usual. Check back more of your marginal hands.",
    ),
    # -- postflop pressure ---------------------------------------------------
    "overfold_flop": Entry(
        behaviour="They fold the flop whenever they miss. They call preflop "
                  "hoping to connect, and when they do not, they are done "
                  "with the hand.",
        why="Any two cards miss the flop most of the time -- that is just how "
            "the deck works. A player who folds every miss is therefore "
            "folding far more often than the price of a bet requires, so your "
            "bet profits regardless of what you hold.",
        do="Bet every flop against them, in position or out. Keep it small, "
           "around a third to half the pot: you are charging them for a fold, "
           "not building a pot, and a small bet risks less to win the same "
           "amount.",
        dont="Do not fire a second barrel on the same automatic basis. Their "
             "flop call means they actually have something, and the turn is a "
             "different decision. Bet the flop always, the turn selectively.",
    ),
    "overfold_turn": Entry(
        behaviour="They call the flop with hands that hoped to improve, then "
                  "fold the turn when the card does not help them.",
        why="This is usually the most valuable leak on the list, because turn "
            "pots are big. They arrive on the turn with a range full of "
            "marginal calls, and folding those means folding most of their "
            "range in a pot that is now worth several times the flop bet.",
        do="Fire the second barrel with everything you bet the flop with. "
           "Two-thirds pot or larger -- the pot is big enough that the fold "
           "equity is worth buying, and your hand does not need to be part of "
           "the reason.",
        dont="Do not keep going on the river automatically. A player who "
             "folds turns this often has already shed their weak hands; the "
             "range that calls a turn bet is genuinely strong, and a third "
             "barrel into it is spewing the profit you just made.",
    ),
    "overfold_river": Entry(
        behaviour="They get to the river and cannot call. They talk "
                  "themselves out of hands they held all the way, and fold "
                  "for the last bet.",
        why="By the river their range is defined and often full of hands that "
            "were never going to be good. Folding too often at the point when "
            "the pot is at its biggest is the single most expensive mistake "
            "in poker, and the bluff you make is being paid at the best price "
            "you will ever get.",
        do="Bluff every river you reach with a hand that cannot win a "
           "showdown. Size up -- three-quarters pot or more. At this fold "
           "frequency they are folding to the decision, not to the price, so "
           "you may as well charge them properly for it.",
        dont="Do not bluff the rivers where their hand is obviously strong -- "
             "when they raised the turn, or when the board completed the draw "
             "they were chasing. An overfolder still calls with the top of "
             "their range, and those are the spots where they have it.",
    ),
    "overfold_cbet": Entry(
        behaviour="When the preflop raiser bets the flop, they get out of the "
                  "way. They treat a continuation bet as information rather "
                  "than as something to be tested.",
        why="They are letting the preflop raiser's range represent more than "
            "it holds. A raising range is mostly unpaired cards that missed, "
            "and folding to it that often means paying for the privilege of "
            "seeing a flop and then not using it.",
        do="Raise more hands before the flop against them and bet the flop "
           "every time. The two multiply: a wider raising range wins more "
           "pots outright because they will not defend against the bet that "
           "follows.",
        dont="Do not read their flop call as weakness because they usually "
             "fold. The times they continue are exactly the times they "
             "connected, so slow down rather than assuming the pattern holds.",
    ),
    # -- stickiness ----------------------------------------------------------
    "station_turn": Entry(
        behaviour="They do not fold once they have any piece of the board. "
                  "Middle pair, bottom pair, a gutshot -- if it can win, they "
                  "are calling to find out.",
        why="Bluffing needs folds to profit, and there are none here. But the "
            "same stubbornness is a gift in the other direction: hands you "
            "would normally check for pot control get paid off by a range "
            "that is far too wide, so your thin value bets are worth much "
            "more than they would be against anyone else.",
        do="Value bet relentlessly and thinly. Top pair with a weak kicker is "
           "a three-street value hand against this player. Second pair is "
           "worth a bet. Size up -- they are not folding to the price, so "
           "charge them.",
        dont="Do not bluff. Not on a scare card, not with a busted draw, not "
             "on the river because the board looks dangerous. Every bluff "
             "against a station is a donation, and the discipline to check "
             "back air is where the profit against them actually comes from.",
    ),
    "station_river": Entry(
        behaviour="They call the last bet with almost anything that beats a "
                  "bluff. Ace high, bottom pair, a busted draw with a high "
                  "card -- they want to see it.",
        why="River calls are where most players are too tight and this one is "
            "too loose. It means your value bets get paid at the biggest pot "
            "of the hand, and it means the bluffs everybody else profits with "
            "are pure losses here.",
        do="Bet any made hand on the river for value, and size up. Hands you "
           "would normally check back -- third pair, ace high on a bricked "
           "board -- become bets, because their calling range is wide enough "
           "to contain worse.",
        dont="Do not raise their river bets light. A player who calls this "
             "wide is not bluffing much themselves; when they lead into you "
             "on the river, they usually have it.",
    ),
    "shows_down": Entry(
        behaviour="They see a lot of showdowns. They want to know what you "
                  "had, and they will pay for the information.",
        why="Reaching showdown that often means calling down with hands most "
            "players fold. The range they arrive with is wide but weak, which "
            "makes marginal value bets profitable and makes bluffing a losing "
            "proposition.",
        do="Value bet thinner than feels comfortable on every street. Hands "
           "you would normally check back for showdown value should bet, "
           "because they will call with worse far more often than usual.",
        dont="Do not try to blow them off a hand with a big bet. Size is not "
             "what makes them fold -- nothing is. A large bluff against a "
             "player who wants to see cards is just a large loss.",
    ),
    "no_showdown": Entry(
        behaviour="They rarely reach a showdown. Somewhere on every street "
                  "they find a reason to let the hand go.",
        why="A player who folds somewhere in every hand is folding too often "
            "somewhere. It does not matter which street they break on -- the "
            "money comes from applying pressure until they do, and cheap "
            "bluffs are profitable against anyone looking for a way out.",
        do="Barrel. Pick the street where they seem least comfortable -- "
           "usually the turn -- and bet it consistently. Small sizes work "
           "fine; they are folding to the fact of the bet, not to its size.",
        dont="Do not assume this means they never have a hand. Their rare "
             "showdowns are strong precisely because everything weak got "
             "folded along the way. When they call two streets, believe them.",
    ),
    # -- their aggression ----------------------------------------------------
    "cbets_always": Entry(
        behaviour="They bet the flop every time they raised before it, "
                  "whether or not the board helped them.",
        why="Their betting range is their entire range, which means it is "
            "mostly air. A bet that means nothing can be attacked directly -- "
            "and because they bet everything, they cannot profitably continue "
            "against a raise with most of it.",
        do="Check-raise them wide on the flop, and float in position with any "
           "backdoor equity to take the pot on the turn when they give up. "
           "Marginal hands play better as raises than as calls here, because "
           "the fold equity is real.",
        dont="Do not do this every single time. A player who bets that often "
             "usually has a strong hand somewhere in the range, and running "
             "the same play repeatedly is how you find it with the worst of "
             "it. Pick your boards -- the ones that miss a raising range.",
    ),
    "cbets_never": Entry(
        behaviour="They raise before the flop and then check when they miss. "
                  "Their check is an announcement that the flop did not help.",
        why="A preflop raiser who checks the flop has given up on the hand. "
            "The pot is sitting there with nobody defending it, and a small "
            "bet takes it far more often than it needs to in order to profit.",
        do="Bet whenever they check to you, in position or out, with any two "
           "cards. Keep it small -- a third of the pot is plenty against a "
           "range that has already surrendered.",
        dont="Do not keep barrelling if they call the stab. A player who "
             "checks their misses has a range full of real hands when they do "
             "continue, so one bet is the play, not three.",
    ),
    "never_check_raises": Entry(
        behaviour="They do not check-raise. When they check, they intend to "
                  "call or fold, never to attack.",
        why="This removes the only real risk of betting into them. Normally a "
            "bet has to account for getting raised off your equity; against "
            "this player it does not, so your entire range can bet whenever "
            "they check.",
        do="Bet every flop they check to you, including hands you would "
           "usually check back for pot control. You will get to see the turn "
           "on your terms every time.",
        dont="Do not extend the assumption to their leads and turn raises. "
             "The aggression they do have is concentrated in those actions "
             "precisely because they never check-raise, so those are strong.",
    ),
    "barrels_relentlessly": Entry(
        behaviour="They keep betting -- flop, turn, and often the river -- at "
                  "a rate no hand-reading can justify. They apply pressure by "
                  "default rather than by plan.",
        why="They cannot possibly have a strong hand often enough to support "
            "that many bets. Most of what they are betting is weak, which "
            "means the hands you would normally fold are now profitable "
            "calls -- they are bluffing into you far more than the pot odds "
            "require.",
        do="Call down lighter, and let them keep firing. Middle pair becomes "
           "a call on two streets. Position helps: flat rather than raise, so "
           "they keep bluffing into a hand that is beating them.",
        dont="Do not try to out-bluff them or raise as a bluff -- you are "
             "attacking the one part of their game that already works. And do "
             "not turn every marginal hand into a hero call; they still have "
             "value hands, and calling three big bets with ace high is not "
             "what this read means.",
    ),
    "three_bets_light": Entry(
        behaviour="They re-raise before the flop constantly, at a frequency "
                  "no value range can support.",
        why="A re-raising range that wide is mostly hands that cannot handle "
            "pressure. They are relying on you folding, which means the "
            "counter is simply to stop folding -- and to occasionally raise "
            "back, since most of their range has to give up.",
        do="Call their re-raises far wider in position, and four-bet light "
           "from time to time. Hands that play well post-flop are better "
           "calls than hands that need to hit -- you want to still be there "
           "on the flop when they have nothing.",
        dont="Do not play a huge pot out of position with a marginal hand "
             "just because you know they are wide. Wide is not the same as "
             "always weak, and out of position a marginal hand is hard to "
             "realise. Fold those and pick your spots in position.",
    ),
    "bluffs_rivers": Entry(
        behaviour="Their river bets are often nothing. They fire the last "
                  "bullet with busted draws and hands that cannot win any "
                  "other way.",
        why="Bluffing the river only works if you get folds. Bluffing it too "
            "often means your bluff-catchers are being paid: a hand that beats "
            "only a bluff becomes a profitable call, which flips the most "
            "expensive decision in poker in your favour.",
        do="Call rivers far wider than normal. Any pair, sometimes ace high. "
           "Just call -- do not raise, because a raise folds out the bluffs "
           "you are trying to collect.",
        dont="Do not extend this to their raises. Bluffing a river bet is "
             "common; raising the river as a bluff is rare even among "
             "aggressive players. If they raise, fold your bluff-catchers.",
    ),
    "light_calls": Entry(
        behaviour="They show up at showdown with hands that had no business "
                  "calling. They pay to see it and then pay again.",
        why="Their calling range is far wider than the price justifies, so "
            "the value of your bets goes up on every street -- particularly "
            "the thin ones you would normally check.",
        do="Value bet every made hand on every street, and size up on the "
           "river. Second pair is a bet. Ace high is sometimes a bet.",
        dont="Do not bluff. They are calling with hands that beat nothing, "
             "which means they are also calling with hands that beat your "
             "bluff.",
    ),
    # -- timing --------------------------------------------------------------
    "tank_folds": Entry(
        behaviour="Long pauses before folding. When they tank and then act, "
                  "the pause itself is the information.",
        why="Somebody genuinely deciding is somebody close to folding -- a "
            "player with a strong hand knows what they are doing quickly. So "
            "the pause tells you the pressure is working, and the quick "
            "actions tell you it is not.",
        do="When they take a long time and then call, respect it: they found "
           "a reason and it is usually a real hand. When they act instantly, "
           "treat it as the top or bottom of their range, not the middle.",
        dont="Do not treat online timing as gospel. Disconnections, phones, "
             "and multi-tabling produce pauses that mean nothing. Use it to "
             "break ties on close decisions, never as the whole basis of one.",
    ),
    "snap_calls": Entry(
        behaviour="Instant calls. No deliberation, no consideration of "
                  "folding -- the call was decided before you bet.",
        why="An instant call is a hand that was never folding but also never "
            "raising: a medium-strength hand, live but beatable. It is almost "
            "never a trap, because a strong hand takes a moment to consider "
            "raising.",
        do="Keep barrelling. Their snap-call range is full of hands that "
           "cannot stand another bet, and a scare card on the turn or river "
           "gives you a genuine chance to move them off it.",
        dont="Do not read every fast action the same way. Instant calls are "
             "weak; instant raises are strong. And in a fast-moving game some "
             "players simply act quickly with everything, which makes the tell "
             "worthless -- check whether their timing varies at all.",
    ),
}


def entry_for(leak_id: str) -> Entry | None:
    return PLAYBOOK.get(leak_id)


# ---------------------------------------------------------------------------
# combinations
# ---------------------------------------------------------------------------
# A leak described on its own understates the case. Two leaks that point the
# same way multiply: a player who folds too much on the flop *and* never
# check-raises has removed both the reason to fear betting and the cost of
# being wrong, and the right play against the pair is more aggressive than the
# right play against either one.
#
# These are the pairs worth spelling out. Deliberately hand-written and
# deliberately few -- a combinatorial explosion of generated pairings would
# bury the two or three that actually change how you play.


@dataclass(frozen=True)
class Combination:
    leaks: frozenset
    headline: str
    body: str


COMBINATIONS: tuple[Combination, ...] = (
    Combination(
        frozenset({"overfold_flop", "never_check_raises"}),
        "Betting the flop against them is close to free",
        "They fold flops too often and they never check-raise, so a flop bet "
        "has almost no downside and wins the pot outright most of the time. "
        "Bet every single flop -- there is no hand you should be checking back "
        "for protection, because nothing bad happens when you bet.",
    ),
    Combination(
        frozenset({"overfold_turn", "overfold_river"}),
        "One bluff carried across two streets wins almost uncontested",
        "They fold both the turn and the river too often, which means a bluff "
        "does not have to work on the street you start it. Bet the turn, and "
        "bet again on the river when they call -- the second barrel gets there "
        "against a range that already folded everything it was comfortable "
        "folding.",
    ),
    Combination(
        frozenset({"station_turn", "station_river"}),
        "A pure value opponent -- stop bluffing entirely",
        "They will not fold on either of the two streets where the pot is "
        "biggest. Against this player there is no bluffing strategy at all: "
        "every chip you make comes from betting hands that are ahead, and the "
        "single biggest improvement you can make is checking back every hand "
        "that cannot call a raise.",
    ),
    Combination(
        frozenset({"folds_blinds", "no_defend"}),
        "Their blinds are yours for the taking",
        "They neither defend their big blind nor fight for it from the small. "
        "Raise every single time it folds to you in late position, at the "
        "minimum size that gets the job done. This adds up faster than any "
        "postflop edge because it happens on every orbit.",
    ),
    Combination(
        frozenset({"barrels_relentlessly", "bluffs_rivers"}),
        "Let them bluff into you on every street",
        "They fire too often on the turn and their river bets are frequently "
        "nothing. The counter is entirely passive: call with your "
        "bluff-catchers and let them keep betting. Raising costs you money "
        "here, because it folds out exactly the hands you are trying to "
        "collect from.",
    ),
    Combination(
        frozenset({"no_three_bet", "folds_to_three_bet"}),
        "Open everything, then believe them",
        "They rarely re-raise and they fold their own opens to a re-raise. "
        "Raise far more hands than usual against them, and re-raise their "
        "opens light. When they do put in a re-raise of their own, it is the "
        "top of a very tight range -- fold anything marginal without a second "
        "thought.",
    ),
    Combination(
        frozenset({"cbets_always", "barrels_relentlessly"}),
        "Their aggression is automatic, not chosen",
        "They bet the flop with everything and keep betting the turn. That is "
        "not a strategy, it is a habit, and it means their betting range is "
        "the whole range on both streets. Call down lighter than feels right, "
        "and check-raise the flop with hands that can stand a re-raise.",
    ),
    Combination(
        frozenset({"limps", "overfold_flop"}),
        "Raise their limps and take the flop",
        "They limp in with weak hands and then fold the flop when they miss. "
        "This is the cheapest money in the game: raise every limp, bet every "
        "flop, and give up the moment they do anything other than fold.",
    ),
    Combination(
        frozenset({"shows_down", "light_calls"}),
        "A calling machine -- value bet everything",
        "They reach showdown constantly and arrive with weak hands. Bet every "
        "made hand on every street, size up on the river, and never bluff. "
        "Hands you would normally check for pot control are pure value here.",
    ),
    Combination(
        frozenset({"cbets_never", "never_check_raises"}),
        "Every pot they do not bet is available",
        "When they check they have given up, and they will not raise you off "
        "the hand. Stab at every checked pot with any two cards, in or out of "
        "position, and fold cheaply the rare times they come back at you.",
    ),
)


def combinations_for(leak_ids) -> list[Combination]:
    """Combinations whose leaks are all present, biggest first."""
    present = set(leak_ids)
    hits = [c for c in COMBINATIONS if c.leaks <= present]
    hits.sort(key=lambda c: -len(c.leaks))
    return hits
