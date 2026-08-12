"""Optional plain-English summary from a local language model.

Everything else in this project is deterministic, and that is a feature: the
same hands always produce the same read, and no number on screen came from
anywhere but the arithmetic. This module is the one exception, so it is fenced
in accordingly.

**It is off unless asked for.** No key, no model, no network call, no import of
anything outside the standard library. With nothing configured the tool behaves
exactly as it did before.

**It talks to any OpenAI-compatible ``/chat/completions`` endpoint.** A local
Ollama is the default because it is free, offline, and keeps hand histories on
the machine that recorded them. A hosted free tier works the same way by
pointing ``VILLAIN_LLM_URL`` at it -- at the cost of sending opponent profiles
to somebody else, which is worth knowing before you switch.

**Credentials live outside the repository.** Settings are read from the
environment, falling back to ``~/.villain/env``, which is deliberately *not* in
the project directory: a key that never sits under the working tree cannot be
committed by an absent-minded ``git add -A``. The file is a plain list of
``NAME=value`` lines and should be readable only by its owner.

**It may not invent numbers.** The model is given a fact sheet built from the
computed profile and asked to explain it; the output is then checked, and any
figure that does not appear in the facts causes the whole response to be
discarded in favour of the static text. A model that rounds 51% to "about half"
is fine; a model that decides they fold 70% is not, and there is no way to tell
which happened by reading the prose. So the check is mechanical.

The value this adds over :mod:`villain.playbook` is synthesis: joining several
findings into one paragraph about this specific player. It is a nicety on top
of the written playbook, never a replacement for it.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "llama3.2"
TIMEOUT = 60

#: Outside the project directory on purpose -- see the module docstring.
CONFIG_PATH = Path.home() / ".villain" / "env"

SETTINGS = ("VILLAIN_LLM_URL", "VILLAIN_LLM_MODEL", "VILLAIN_LLM_KEY")


def _config() -> dict[str, str]:
    """Settings from ``~/.villain/env``. The environment always wins."""
    values: dict[str, str] = {}
    try:
        text = CONFIG_PATH.read_text()
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name in SETTINGS:
            values[name] = value.strip().strip("\'\"")
    return values


def setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _config().get(name) or default

SYSTEM = (
    "You are a poker coach writing a scouting report on one opponent, for a "
    "player about to sit down with them. You are given facts already computed "
    "from their hand history. Write a detailed profile.\n\n"
    "Cover, in this order, as flowing paragraphs:\n"
    "1. What kind of player they are -- the shape of their game, not a "
    "restatement of the label.\n"
    "2. Their strengths: what they do competently, and which parts of their "
    "game not to attack. If the facts show none worth naming, say so rather "
    "than inventing one.\n"
    "3. Their exploitable weaknesses, most valuable first, and the mechanism "
    "behind each -- why it costs them money, not merely that it exists.\n"
    "4. What to do: concrete adjustments, with streets and bet sizes where the "
    "facts supply them.\n"
    "5. What not to do: how a player who read this profile would over-adjust "
    "and hand the money back. This matters as much as the section before it.\n\n"
    "Rules you must follow:\n"
    "- Use only the facts given. Never state a figure that is not in them. To "
    "describe a frequency you were not told, use words or leave it out.\n"
    "- Prefer words to numbers throughout. \'folds most rivers\' beats \'51%\'.\n"
    "- Four to six paragraphs, at least 250 words. No headings, no bullet "
    "points, no preamble, no sign-off. Just the report.\n"
    "- Plain English a competent player would use at the table.\n"
    "- Be specific to this opponent. Generic advice that would fit anyone is "
    "worthless here.\n"
    "- If the sample is small, say which parts of the read are provisional "
    "rather than hedging every sentence."
)


@dataclass
class Narration:
    text: str
    model: str
    endpoint: str


class Unavailable(RuntimeError):
    """No model configured, or it could not be reached."""


def enabled() -> bool:
    """True when a narrator has been explicitly configured."""
    return bool(setting("VILLAIN_LLM_MODEL") or setting("VILLAIN_LLM_URL"))


def describe_endpoint() -> str:
    """Where calls go, for showing the user without leaking the key."""
    url = setting("VILLAIN_LLM_URL", DEFAULT_URL)
    host = url.split("/")[2] if "//" in url else url
    return f"{setting('VILLAIN_LLM_MODEL', DEFAULT_MODEL)} at {host}"


def fact_sheet(payload: dict) -> str:
    """The only thing the model is allowed to know, built from the profile."""
    lines = [
        f"Player: {payload.get('name', 'unknown')}",
        f"Table size: {payload.get('table_mix') or payload.get('regime_label') or payload.get('regime', '')}",
        f"Hands observed: {payload.get('hands', 0)} "
        f"({payload.get('sample_quality', 'unknown')})",
        f"Player type: {payload.get('archetype', 'unknown')} "
        f"({round(100 * payload.get('archetype_confidence', 0))}% confident)",
        f"Type description: {payload.get('summary', '')}",
        f"Skill rating: {payload.get('skill', {}).get('score', 0)} out of 100 "
        f"({payload.get('skill', {}).get('tier', 'unknown')})",
    ]
    for item in payload.get("strengths") or []:
        lines.append(f"Does competently (do not attack): {item}")

    # Only well-observed frequencies. A stat with four observations is noise,
    # and handing it to a model invites a confident sentence about nothing.
    measured = []
    for stat, entry in (payload.get("stats") or {}).items():
        if isinstance(entry, dict) and (entry.get("opportunities") or 0) >= 10:
            measured.append(f"- {stat}: {round(100 * entry['value'])}% over "
                            f"{round(entry['opportunities'])} spots")
    if measured:
        lines.append("Measured frequencies, with the observations behind each:")
        lines.extend(measured)

    if payload.get("leaks"):
        lines.append("Leaks found, most valuable first:")
        for leak in payload["leaks"]:
            lines.append(
                f"- {leak['headline']}. {leak['in_words']} "
                f"Worth about {leak['severity_bb100']} big blinds per 100 hands "
                f"({leak['size']}, {leak['tier']} read). "
                f"What they are doing: {leak['behaviour']} "
                f"Do: {leak['do']} Do not: {leak['dont']}")
    else:
        lines.append("No leak has enough evidence behind it yet.")
    for combo in payload.get("combinations", []):
        lines.append(f"Compounding: {combo['headline']}. {combo['body']}")
    return "\n".join(lines)


def narrate(payload: dict, *, url: str | None = None, model: str | None = None,
            timeout: int = TIMEOUT) -> Narration:
    """Ask the configured model to summarise a profile. Raises on any problem."""
    url = url or setting("VILLAIN_LLM_URL", DEFAULT_URL)
    model = model or setting("VILLAIN_LLM_MODEL", DEFAULT_MODEL)
    facts = fact_sheet(payload)

    body = json.dumps({
        "model": model,
        "temperature": 0.3,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": facts}],
    }).encode()
    headers = {"Content-Type": "application/json"}
    key = setting("VILLAIN_LLM_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        # The endpoint answered and refused. Its own message is more useful
        # than anything guessed here.
        detail = ""
        try:
            detail = json.load(exc).get("error", {}).get("message", "")
        except Exception:
            pass
        raise Unavailable(
            f"{model} returned {exc.code}"
            + (f": {detail}" if detail else f" ({exc.reason})")
            + (". This is usually temporary -- try again." if exc.code >= 500 else "")
        ) from exc
    except urllib.error.URLError as exc:
        # Nothing answered at all. Only suggest starting a local model when the
        # endpoint actually is local; telling somebody to "ollama pull" a
        # hosted model name is worse than saying nothing.
        local = "localhost" in url or "127.0.0.1" in url
        hint = (f"Start it with 'ollama serve' and 'ollama pull {model}'."
                if local else
                "Check the endpoint and your network, or set VILLAIN_LLM_URL "
                "to another OpenAI-compatible endpoint.")
        raise Unavailable(f"could not reach {url}: {exc.reason}. {hint}") from exc
    except (TimeoutError, OSError) as exc:
        raise Unavailable(f"model at {url} did not respond: {exc}") from exc

    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise Unavailable(f"unexpected response from {url}") from exc

    invented = unsupported_numbers(text, facts)
    if invented:
        raise Unavailable(
            f"model stated figures that are not in the data ({', '.join(invented)}); "
            "discarded")
    return Narration(text=text, model=model, endpoint=url)


#: Numbers a sentence can contain without claiming anything about the player:
#: counting words rendered as digits, and the streets of a hand.
SAFE_NUMBERS = {"0", "1", "2", "3", "4", "5", "100"}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def unsupported_numbers(text: str, facts: str) -> list[str]:
    """Figures in ``text`` that do not appear in ``facts``.

    Deliberately strict. A model that invents a fold frequency produces prose
    indistinguishable from a correct one, so the only defence is refusing to
    show any figure the arithmetic did not produce.
    """
    known = set(_NUMBER.findall(facts))
    # A percentage stated as "51" is supported by a fact sheet saying "51%",
    # and rounding to a whole number is fine.
    for value in list(known):
        if "." in value:
            known.add(value.split(".")[0])
            try:
                known.add(str(round(float(value))))
            except ValueError:
                pass
    out = []
    for value in _NUMBER.findall(text):
        if value in known or value in SAFE_NUMBERS:
            continue
        out.append(value)
    return sorted(set(out))
