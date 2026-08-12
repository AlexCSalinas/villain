"""Plain-language definitions for everything the interface shows.

A profiler is useless if the reader has to already know what "fold vs turn bet,
47%, breakeven 40%" implies. Every statistic here carries three sentences: what
the number counts, what a high one means for you, and what a low one means --
because for most of these, both directions are exploitable and they call for
opposite play. Overfolding says bluff more; underfolding says stop bluffing and
value bet. Getting that backwards costs more than not knowing the number.

Kept as data rather than prose in the template so the same words appear in the
UI, in exports, and in any future surface, and so a test can assert that no
statistic reaches the screen without an explanation.
"""

from __future__ import annotations

#: stat -> what it counts, what high means, what low means.
STATS: dict[str, dict[str, str]] = {
    "vpip": {
        "what": "How often they voluntarily put money in before the flop — any "
                "call or raise, but not a blind they were forced to post.",
        "high": "Playing too many hands. Their range is full of weak holdings, "
                "so they will miss most flops.",
        "low": "Very selective. When they enter a pot, respect it — but their "
               "blinds are free money.",
    },
    "pfr": {
        "what": "How often they raise before the flop.",
        "high": "Aggressive preflop. Expect to be raised, and defend wider.",
        "low": "Passive entry — they call more than they raise, which caps how "
               "strong their range can be.",
    },
    "three_bet": {
        "what": "When someone raises in front of them, how often they re-raise.",
        "high": "Re-raising more often than a value range allows, so many are "
                "bluffs. Fight back wider.",
        "low": "Their re-raises are the top of their range. Fold anything "
               "marginal to one.",
    },
    "fold_to_three_bet": {
        "what": "After raising, how often they fold to a re-raise.",
        "high": "Their opening range collapses under pressure. Re-raise them light.",
        "low": "They defend their raises. Re-raise for value, not as a bluff.",
    },
    "bb_defend": {
        "what": "In the big blind facing a raise, how often they play rather "
                "than fold.",
        "high": "Hard to steal from. Open a tighter range against them.",
        "low": "Giving up their blind. Raise every button — the folds alone pay.",
    },
    "limp": {
        "what": "How often they enter the pot by calling the big blind instead "
                "of raising, when first to act.",
        "high": "A limping range is capped and passive. Raise over the top and "
                "bet the flop.",
        "low": "Normal — good players rarely limp.",
    },
    "fold_to_steal": {
        "what": "In the blinds facing a late-position raise, how often they fold.",
        "high": "Their blinds are free. Open every button and small blind.",
        "low": "They defend. Steal with hands that can play after the flop.",
    },
    "cbet:flop": {
        "what": "Having raised before the flop, how often they bet the flop.",
        "high": "They bet their whole range, so most of it is air. Check-raise "
                "and float wide.",
        "low": "A checked flop means they missed. Take the pot away.",
    },
    "cbet:turn": {
        "what": "Having bet the flop, how often they fire again on the turn.",
        "high": "They keep barrelling with too much. Call down lighter.",
        "low": "They give up on the turn. Float the flop and take it next street.",
    },
    "fold_vs_bet:flop": {
        "what": "Facing a bet on the flop, how often they fold.",
        "high": "Bet every flop against them — the folds alone show a profit.",
        "low": "They will not fold flops. Bet for value, not as a bluff.",
    },
    "fold_vs_bet:turn": {
        "what": "Facing a bet on the turn, how often they fold.",
        "high": "The most profitable leak there is — turn pots are big. Fire a "
                "second barrel with anything.",
        "low": "Sticky on the turn. Value bet thinner and abandon bluffs.",
    },
    "fold_vs_bet:river": {
        "what": "Facing a bet on the river, how often they fold.",
        "high": "Bluff every river you get to, and size up.",
        "low": "A calling station on the river. Bet made hands, never bluff.",
    },
    "fold_to_cbet:flop": {
        "what": "Facing a bet from the player who raised before the flop, how "
                "often they fold.",
        "high": "They surrender the flop. Raise more hands and bet every one.",
        "low": "They stick around. Continue with real equity, not automatically.",
    },
    "fold_to_cbet:turn": {
        "what": "Facing a second barrel from the preflop raiser, how often they fold.",
        "high": "The second barrel prints. Keep firing.",
        "low": "They call down. Barrel with hands that improve, not with air.",
    },
    "check_raise:flop": {
        "what": "Having checked the flop, how often they raise a bet.",
        "high": "Dangerous to bet into. Check back more of your marginal hands.",
        "low": "Their check is a surrender. Bet every flop they check to you.",
    },
    "donk:flop": {
        "what": "Betting into the previous street's aggressor instead of checking "
                "to them.",
        "high": "Unusual and usually unbalanced — often a weak made hand "
                "protecting itself.",
        "low": "Normal. They check to the raiser as expected.",
    },
    "wtsd": {
        "what": "Having seen a flop, how often they reach showdown.",
        "high": "They want to see cards. Value bet thin; bluffing is throwing "
                "money away.",
        "low": "They are looking for a reason to fold. Barrel relentlessly.",
    },
    "wsd": {
        "what": "Of the showdowns they reach, how often they win.",
        "high": "They get there with real hands. Their calls mean something.",
        "low": "They arrive with weak holdings — paying off too often.",
    },
    "wwsf": {
        "what": "Having seen a flop, how often they end up winning the pot.",
        "high": "They take their share of pots after the flop.",
        "low": "They see flops and give up. Bet at them.",
    },
    "aggression:flop": {
        "what": "Of everything they do on the flop, how much is betting or raising.",
        "high": "Constant pressure. Let them bet your good hands for you.",
        "low": "Passive — they check and call. Bet your value hands relentlessly.",
    },
    "aggression:turn": {
        "what": "Of everything they do on the turn, how much is betting or raising.",
        "high": "Bets more turns than any range supports. Call down lighter.",
        "low": "Gives up the turn. Take the pot when they check.",
    },
    "aggression:river": {
        "what": "Of everything they do on the river, how much is betting or raising.",
        "high": "Bluffs rivers often. Bluff-catchers are printing — just call.",
        "low": "Only bets rivers with the goods. Fold to their river bets.",
    },
    "river_bet_bluff": {
        "what": "Of the river bets where they later showed, how many were weak "
                "hands.",
        "high": "They bluff rivers a lot. Call wider.",
        "low": "River bets are value. Believe them.",
    },
    "sd_light_call": {
        "what": "How often they call down and show a weak hand.",
        "high": "They pay off. Value bet every made hand.",
        "low": "Their calls are strong. Do not bet thin against them.",
    },
    "tank_fold": {
        "what": "How often a fold came after a long pause.",
        "high": "Long pauses mean they are looking for a fold. Their quick "
                "actions are the strong ones.",
        "low": "They act at a steady speed — less timing information available.",
    },
    "snap_call": {
        "what": "How often a call came instantly.",
        "high": "Instant calls are weak-but-live hands, never traps. Keep betting.",
        "low": "They think before calling — timing tells you less.",
    },
}

#: Words the interface uses that carry a specific meaning.
TERMS: dict[str, str] = {
    "guesswork": "Under 50 hands. Not enough to read anyone — treat everything "
                 "here as a placeholder.",
    "thin": "50 to 150 hands. Preflop numbers are becoming real; anything about "
            "turns and rivers is still mostly guesswork.",
    "usable": "150 to 500 hands. Preflop reads are reliable and the bigger "
              "postflop leaks will show.",
    "solid": "Over 500 hands. Trust it, including the finer postflop numbers.",
    "watch": "Seen, but not confirmed. Probably real and not yet worth acting "
             "on -- no price is given because the tool is not confident enough "
             "to tell you what it is worth. Keep playing them and it will "
             "either firm up or disappear.",
    "tentative": "The evidence leans this way but could still be luck. Worth "
                 "knowing, not worth changing your whole game for.",
    "likely": "Probably real. Act on it, and keep watching.",
    "strong": "The sample supports this. Attack it.",
    "bb/100": "Big blinds won per 100 hands — the standard unit of poker "
              "winrate. At 5c/10c stakes, 1 bb/100 is 10 cents per 100 hands.",
    "breakeven": "The frequency at which the exploit stops making money. Bluffing "
                 "a two-thirds pot bet needs them to fold 40% of the time to "
                 "break even, so anything above that is profit.",
    "field": "What a typical player at this table size does — context for "
             "whether a number is unusual, not a target to aim at.",
    "estimate": "The frequency after accounting for sample size. A player who "
                "folded 3 of 3 is not a 100% folder, and this pulls that back "
                "toward reality.",
    "95% range": "Where their true frequency probably sits. A wide range means "
                 "not enough hands yet.",
    "available": "Roughly what a perfect opponent could win from their leaks, in "
                 "big blinds per 100 hands, if they attacked every one.",
    "confidence": "How much of this read comes from their actual hands rather "
                  "than from assumptions about players in general.",
}

#: What a low score in each rated area means for you. The rating knows these
#: things whether or not a statistical test clears, and saying nothing about
#: them makes a weak player look unreadable.
COMPONENTS: dict[str, str] = {
    "hand selection": "They play the wrong hands -- too many, too few, or "
                      "entering pots by calling. Punish it before the flop by "
                      "raising more of your own hands against them.",
    "preflop aggression": "They call where they should raise. Their calling "
                          "range is capped, so bet at them after the flop and "
                          "believe them when they finally raise.",
    "postflop aggression": "Their betting after the flop is off -- too passive "
                           "to protect their good hands, or too busy to have "
                           "them. Either way their bets and checks say more "
                           "than they should.",
    "discipline vs bets": "They fold at the wrong frequencies when facing "
                          "bets. Whichever way they err, the answer is to "
                          "bet more or bluff less accordingly.",
    "showdown judgement": "They arrive at showdown with the wrong hands -- "
                          "paying off too often, or folding hands that were "
                          "good. Value bet thinner against them.",
    "bet sizing": "Their sizes are readable or badly chosen. One size for "
                  "every situation means the size tells you nothing they meant "
                  "it to, and often a lot they did not.",
    "resistance to exploitation": "How much money the leaks found against them "
                                  "are worth in total.",
}


def component_help(name: str) -> str | None:
    return COMPONENTS.get(name)


#: How the table-size split is explained.
REGIMES: dict[str, str] = {
    "hu": "Heads-up. Everyone plays far more hands here, so a 55% VPIP is tight.",
    "3max": "Three-handed. You are in a blind two hands out of three.",
    "6max": "Six-handed. The standard short table.",
    "full": "Seven or more players. The tightest of the four.",
}


def stat_help(stat: str) -> dict[str, str] | None:
    """Explanation for a stat, falling back to the street-agnostic version."""
    if stat in STATS:
        return STATS[stat]
    base = stat.rsplit(":", 1)[0]
    return STATS.get(base)


def payload() -> dict:
    return {"stats": STATS, "terms": TERMS, "regimes": REGIMES}
