"""Every known player, ranked."""

from __future__ import annotations

import gzip
import json
import secrets
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..analyze import as_dict, enrich
from ..archetypes import ARCHETYPE_BY_NAME, deviations
from ..db import DEFAULT_PATH, Store, split_key
from ..dynamics import adjustments
from ..exploits import RULES, find_watchlist
from ..features import record_hands
from ..evidence import find as find_evidence
from ..glossary import payload as glossary_payload, stat_help
from ..model import hand_from_dict, hand_to_dict
from ..identity import askable_questions, auto_answers, session_questions, suggest_links
from ..skill import weaknesses
from ..narrate import Unavailable, enabled as narrator_enabled, narrate
from ..parsers import UnknownFormat, parse_file
from ..priors import population_mean
from ..profile import build_profiles, build_unified, primary_regime
from ..stats import VS_HERO
from ..timing import timing_tells
from ..replay import replay

from .payloads import roster_payload

def leaderboard_payload(store: Store) -> dict:
    """Every known player, ranked.

    Two orderings matter and they are not the same question. Skill answers
    "who is dangerous"; attackable bb/100 answers "who is worth sitting with".
    A competent player with one exploitable habit can be worth more to you than
    a weak player you have barely seen, so both are shown and the table sorts
    on either.
    """
    ranked = roster_payload(store)
    return {"players": sorted(ranked, key=lambda r: -r["skill"])}


# ---------------------------------------------------------------------------
# hero: what only your own hand history can show
# ---------------------------------------------------------------------------
# Grading every fold means fitting the population hand-strength model first
# (villain.reads.fit) and walking hero's several thousand hands through the
# 7-card evaluator -- tens of seconds on a database this size, and unchanged
# from one request to the next unless new hands were imported. An in-memory
# cache alone only pays that once *per running server*, and this UI gets
# stopped and restarted often -- so the finished payload (JSON-safe: no
# sklearn object in it) is also persisted next to the database, keyed by hand
# count rather than time, the same as the in-memory layer. The model itself
# is cheap to refit inside one process and expensive to pickle safely across
# versions, so only the memory layer holds it.
