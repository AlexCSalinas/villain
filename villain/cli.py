"""Command line interface.

    villain import <files...>        read hand histories into the database
    villain players                  who is in the database
    villain profile <name>           the full read on somebody
    villain scout <file>             profile a file without storing it
    villain link --suggest           find accounts that may be one person
    villain ui                       open the local web interface
    villain fit                      re-estimate priors from your own games
    villain rebuild                  recompute every profile from stored hands
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import as_dict
from .db import DEFAULT_PATH, ImportReport, Store
from .features import record_hands
from .identity import suggest_links
from .parsers import parse_paths
from .profile import build_unified
from .report import profile_card, roster
from .skill import leaderboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="villain", description="Profile poker opponents from hand histories.")
    parser.add_argument("--db", type=Path, default=DEFAULT_PATH,
                        help=f"database path (default {DEFAULT_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="read hand histories into the database")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("players", help="list known players")
    p.add_argument("--min-hands", type=int, default=1)

    p = sub.add_parser("profile", help="the full read on a player")
    p.add_argument("name")
    p.add_argument("--by-table", action="store_true",
                   help="split the profile by table size instead of pooling it")
    p.add_argument("--regime", help="with --by-table: hu, 3max, 6max or full")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--narrate", action="store_true",
                   help="add a plain-English summary from a local model "
                        "(needs VILLAIN_LLM_MODEL; see README)")

    p = sub.add_parser("scout", help="profile a file without storing it")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--min-hands", type=int, default=20)
    p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("link", help="merge two accounts belonging to one person")
    p.add_argument("keep", nargs="?", help="player id to keep")
    p.add_argument("absorb", nargs="?", help="player id to fold into it")
    p.add_argument("--suggest", action="store_true")

    p = sub.add_parser("fit", help="learn from everything in the database")
    p.add_argument("--min-players", type=int, default=8)

    sub.add_parser("rebuild", help="recompute all profiles from stored hands")

    p = sub.add_parser("ui", help="serve the local web interface")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")

    p = sub.add_parser("note", help="attach a note to a player")
    p.add_argument("name")
    p.add_argument("body", nargs="+")

    args = parser.parse_args(argv)
    handler = globals()[f"_cmd_{args.command}"]
    return handler(args)


# ---------------------------------------------------------------------------


def _cmd_import(args) -> int:
    report = ImportReport()
    with Store(args.db) as store:
        for path, hands in parse_paths(args.paths):
            report.files += 1
            if not args.quiet:
                print(f"  {path.name}: {len(hands)} hands")
            store.add_hands(hands, report)
    if report.files == 0:
        print("No file matched a known format.", file=sys.stderr)
        return 1
    print(report)
    return 0


def _cmd_players(args) -> int:
    with Store(args.db) as store:
        rows = [r for r in store.players() if (r["hands"] or 0) >= args.min_hands]
        if not rows:
            print("No players yet. Start with: villain import <file>")
            return 0
        print(f"{'id':>4s}  {'player':18s} {'hands':>6s}  aliases")
        for row in rows:
            print(f"{row['id']:>4d}  {row['display_name'][:18]:18s} "
                  f"{row['hands'] or 0:6d}  {row['aliases'] or ''}")
    return 0


def _cmd_profile(args) -> int:
    with Store(args.db) as store:
        matches = store.find_player(args.name)
        if not matches:
            print(f"No player matching {args.name!r}.", file=sys.stderr)
            return 1
        if len(matches) > 1:
            print("Several players match:", file=sys.stderr)
            for row in matches:
                print(f"  {row['id']}  {row['display_name']}", file=sys.stderr)
            return 1
        player_id = int(matches[0]["id"])
        if args.by_table or args.regime:
            profiles = store.profiles(player_id)
            if args.regime:
                profiles = [p for p in profiles if p.regime == args.regime]
        else:
            unified = store.profile(player_id)
            profiles = [unified] if unified else []
        if not profiles:
            print("No hands recorded for that player and table size.", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps([as_dict(p) for p in profiles], indent=2))
            return 0
        for profile in profiles:
            print(profile_card(profile, verbose=args.verbose))
            if args.narrate:
                print(_narration(profile))
        for note in store.notes(player_id):
            print(f"note: {note['body']}")
    return 0


def _narration(profile) -> str:
    """Plain-English summary, or the reason there isn't one.

    Failures are reported rather than swallowed: the usual cause is that no
    model is running, and saying so is more useful than silently printing
    nothing.
    """
    from .analyze import as_dict
    from .narrate import Unavailable, narrate
    try:
        result = narrate(as_dict(profile))
    except Unavailable as exc:
        return f"IN SHORT: unavailable -- {exc}\n"
    body = "\n".join(f"  {line}" for line in _wrap_plain(result.text, 74))
    return f"IN SHORT  ({result.model})\n{body}\n"


def _wrap_plain(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width) or [""]


def _cmd_scout(args) -> int:
    """Read files straight through to a report, touching no database writes."""
    from .profile import primary_regime

    hands = [h for _, batch in parse_paths(args.paths) for h in batch]
    if not hands:
        print("No file matched a known format.", file=sys.stderr)
        return 1
    books = record_hands(hands)

    fitted: dict[str, dict[str, tuple[float, float]]] = {}
    if Path(args.db).exists():
        with Store(args.db) as store:
            for by in books.values():
                if not by:
                    continue
                home = primary_regime(by)
                if home not in fitted:
                    fitted[home] = store.fitted_priors(home)

    profiles = [
        p for p in (
            build_unified(by, priors=fitted.get(primary_regime(by)) or None)
            for by in books.values() if by
        )
        if p is not None and p.hands >= args.min_hands
    ]
    if not profiles:
        print(f"No player reached {args.min_hands} hands.", file=sys.stderr)
        return 1
    print(f"{len(hands)} hands, {len(profiles)} players\n")
    print(roster(leaderboard(profiles)))
    if args.verbose:
        for profile in profiles:
            print()
            print(profile_card(profile, verbose=True))
    return 0


def _cmd_link(args) -> int:
    with Store(args.db) as store:
        if args.suggest or not (args.keep and args.absorb):
            suggestions = suggest_links(store)
            if not suggestions:
                print("No candidate merges.")
                return 0
            print("Candidate merges (nothing is applied automatically):")
            for s in suggestions:
                print(f"  villain link {s.keep} {s.absorb}"
                      f"   # {s.absorb_name} -> {s.keep_name}, "
                      f"confidence {s.confidence:.0%}")
                print(f"      {s.reason}")
            return 0
        store.link(int(args.keep), int(args.absorb))
        print(f"Merged {args.absorb} into {args.keep} and rebuilt their profile.")
    return 0


def _cmd_fit(args) -> int:
    """Learn what this database can support, and say what it cannot.

    Three models, each gated on having enough data. Reporting the refusals is
    the point: a clustering fitted to six players would look exactly as
    authoritative as one fitted to six hundred.
    """
    from .cluster import NotEnoughData as ClustersNeedMore
    from .cluster import fit_clusters
    from .reads import NotEnoughData as ReadsNeedMore
    from .reads import build_dataset
    from .reads import fit as fit_strength

    with Store(args.db) as store:
        print("priors")
        fitted = store.fit_priors(min_players=args.min_players)
        if fitted:
            for regime, count in sorted(fitted.items()):
                print(f"  {regime}: re-estimated {count} priors from your own games")
        else:
            print(f"  not enough players yet (need {args.min_players} per table size); "
                  "using the built-in population priors")

        print("clusters")
        profiles = [p for row in store.players()
                    for p in store.profiles(int(row["id"]))]
        try:
            model = fit_clusters(profiles)
            print(f"  {model.n_components} groups over {model.trained_on} profiles")
            for cluster in model.clusters:
                print(f"    {cluster.describe()}")
        except ClustersNeedMore as exc:
            print(f"  {exc}")

        print("hand strength")
        rows = build_dataset(store.stored_hands())
        try:
            strength = fit_strength(rows)
            print(f"  trained on {strength.rows} revealed decisions "
                  f"({strength.unbiased_rows} unbiased); "
                  f"out-of-fold error {strength.mae:.3f}")
            for row in store.players():
                read = strength.read(str(row["id"]))
                if read:
                    print(f"    {row['display_name']}: {read}")
        except ReadsNeedMore as exc:
            print(f"  {exc}")
    return 0


def _cmd_rebuild(args) -> int:
    with Store(args.db) as store:
        print(f"Rebuilt {store.rebuild()} player profile(s) from stored hands.")
    return 0


def _cmd_ui(args) -> int:
    from .web import serve
    serve(db=args.db, port=args.port, open_browser=not args.no_browser)
    return 0


def _cmd_note(args) -> int:
    with Store(args.db) as store:
        matches = store.find_player(args.name)
        if len(matches) != 1:
            print(f"Need exactly one match for {args.name!r}.", file=sys.stderr)
            return 1
        store.add_note(int(matches[0]["id"]), " ".join(args.body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
