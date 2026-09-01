"""Ollama client, response cache, and runtime token-match verification.

Conversations are replayed with **prefilled assistant acknowledgements**: every user
turn is followed by a fixed `"Got it."` that the model did not generate. That choice
is what makes the token match provable rather than estimated -- a blocked and an
interleaved conversation become the identical multiset of strings in a different
order, so their prompts tokenise to the same length by construction.

It also has a consequence worth stating plainly rather than burying: the model is not
doing interactive work mid-conversation. This measures extraction and aggregation over
a transcript, not live turn-by-turn state maintenance. The generative variant is named
as next work in the README.

Only the final turn is generated, and only it is schema-constrained.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama
from pydantic import BaseModel, Field

from .generator import Conversation
from .ops import TaskKind, assign_slots, slot_key

ACK = "Got it."

CACHE_DIR = Path("results/cache")


class ModelSpec(BaseModel):
    """Everything that can change a response, recorded so a row is reproducible.

    `digest` comes from `ollama show` and pins the actual weights: a tag like
    `gemma4:12b` can be repointed upstream, and a silently swapped model would look
    like a behavioural finding.
    """

    name: str
    digest: str = ""
    num_ctx: int = 16384
    temperature: float = 0.0
    top_k: int = 1
    seed: int = 0
    num_predict: int = 512
    think: bool = False  # gemma4 reasons by default and returns empty content

    @property
    def options(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_k": self.top_k, "seed": self.seed,
                "num_ctx": self.num_ctx, "num_predict": self.num_predict}


class FinalState(BaseModel):
    """The constrained output schema.

    Historically one field per task KIND, which capped the design at two tasks. The
    schema is now built per task SLOT (`schema_for`), so a conversation with two
    shopping lists asks for `shopping_0` and `shopping_1` separately. These fields
    remain for the two-task case and for backwards compatibility with cached rows.
    """

    shopping: list[str] | None = Field(default=None)
    # Named keys, not positional pairs. The first demo run exposed why: with a bare
    # "array of two strings" the model chose [title, time] in the blocked condition and
    # [time, title] in the interleaved one, and the scorer read the difference as six
    # tracking failures in a conversation the model had tracked perfectly. An ambiguous
    # schema does not just add noise -- it added noise that CORRELATED WITH THE
    # CONDITION, which would have manufactured a switch cost out of nothing.
    schedule: list[dict[str, str]] | None = Field(default=None)

    @staticmethod
    def schema_for(tasks: list[TaskKind]) -> dict[str, Any]:
        """One field per task SLOT, so two shopping lists get two fields.

        Only the slots this conversation actually tracks are requested, so an unasked
        field can never be scored as a hallucination. Field names match the oracle's
        `slot_key`, which is what lets the scorer diff them without a mapping table.
        """
        props: dict[str, Any] = {}
        for t, sl in zip(tasks, assign_slots(tasks)):
            key = slot_key(t, sl)
            if t is TaskKind.SHOPPING:
                props[key] = {"type": "array", "items": {"type": "string"}}
            else:
                props[key] = {"type": "array", "items": {
                    "type": "object",
                    "properties": {"time": {"type": "string"},
                                   "title": {"type": "string"}},
                    "required": ["time", "title"]}}
        return {"type": "object", "properties": props, "required": list(props)}


@dataclass
class RunResult:
    conversation: Conversation
    model: ModelSpec
    raw_final: str
    parsed: FinalState | dict[str, Any] | None
    parse_ok: bool
    n_retries: int
    wall_seconds: float
    prompt_tokens: int = 0
    eval_tokens: int = 0
    constrained: bool = True
    error: str = ""


def build_messages(conv: Conversation) -> list[dict[str, str]]:
    """System prompt, then (user turn, prefilled ack) pairs, then the final request."""
    msgs = [{"role": "system", "content": conv.system}]
    for turn in conv.turns:
        msgs.append({"role": "user", "content": turn})
        msgs.append({"role": "assistant", "content": ACK})
    msgs.append({"role": "user", "content": conv.final_request})
    return msgs


def cache_key(conv: Conversation, model: ModelSpec, constrained: bool) -> str:
    """sha256 over the exact prompt and every generation parameter.

    Keyed on the rendered messages rather than the seed, so a change to any template
    or to the ack string invalidates stale entries instead of silently reusing them.
    """
    payload = json.dumps({
        "messages": build_messages(conv),
        "model": model.name, "digest": model.digest,
        "options": model.options, "think": model.think,
        "constrained": constrained,
        "tasks": [t.value for t in conv.tasks],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_slots(raw: str) -> dict[str, Any] | None:
    """Parse a response into `{slot_key: value}` without a fixed field list.

    `FinalState` cannot express an arbitrary number of slots, so the slot-aware path
    keeps the parsed answer as a plain dict. The scorer compares it against the
    oracle's snapshot, which is keyed the same way.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse(raw: str) -> FinalState | None:
    """Parse a final response into the schema, tolerating fenced or padded JSON.

    Free-form runs (the constrained-vs-unconstrained check) need the tolerance; the
    constrained path should never exercise it.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return FinalState.model_validate_json(text[start:end + 1])
    except Exception:
        return None


def run_conversation(conv: Conversation, model: ModelSpec, constrained: bool = True,
                     use_cache: bool = True, extract: bool = False) -> RunResult:
    """Replay a conversation and generate only its final turn.

    Retries cover *transport* failures only. A parse failure is not retried, because
    decoding is greedy (temperature 0, top_k 1): re-asking would reproduce the same
    tokens exactly, so a retry loop would burn time and inflate no metric. Recording
    parse_ok=False immediately is both faster and more honest.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(conv, model, constrained)
    cached = CACHE_DIR / f"{key}.json"

    if use_cache and cached.exists():
        blob = json.loads(cached.read_text())
    else:
        messages = build_messages(conv)
        fmt = FinalState.schema_for(conv.tasks) if constrained else None
        blob, last_err, retries = None, "", 0
        for attempt in range(3):
            t0 = time.time()
            try:
                kw: dict[str, Any] = {"model": model.name, "messages": messages,
                                      "options": model.options, "think": model.think}
                if fmt is not None:
                    kw["format"] = fmt
                r = ollama.chat(**kw)
                blob = {"content": r["message"]["content"],
                        "prompt_eval_count": r.get("prompt_eval_count", 0),
                        "eval_count": r.get("eval_count", 0),
                        "wall": time.time() - t0, "retries": attempt, "error": ""}
                break
            except Exception as exc:                      # transport / server errors only
                last_err, retries = f"{type(exc).__name__}: {exc}", attempt
                time.sleep(1.0 + attempt)
        if blob is None:
            blob = {"content": "", "prompt_eval_count": 0, "eval_count": 0,
                    "wall": 0.0, "retries": retries, "error": last_err}
        if use_cache and not blob["error"]:
            cached.write_text(json.dumps(blob))

    parsed = parse_slots(blob["content"])
    # Free-form answers are usually correct prose rather than JSON. A second extraction
    # pass turns them into the schema so the comparison is about tracking, not format.
    if parsed is None and extract and not constrained:
        st = extract_state(blob["content"], conv.tasks, model, use_cache)
        parsed = st.model_dump() if isinstance(st, FinalState) else st
    return RunResult(
        conversation=conv, model=model, raw_final=blob["content"], parsed=parsed,
        parse_ok=parsed is not None, n_retries=blob["retries"],
        wall_seconds=blob["wall"], prompt_tokens=blob["prompt_eval_count"],
        eval_tokens=blob["eval_count"], constrained=constrained, error=blob["error"],
    )


EXTRACT_SYSTEM = (
    "You convert an assistant's answer into JSON. Copy the values exactly as written. "
    "Do not add, remove, reorder, correct or infer anything. If the answer omits "
    "something, omit it too."
)


def extract_state(raw: str, tasks: list[TaskKind], model: ModelSpec,
                  use_cache: bool = True) -> FinalState | None:
    """Second-pass extraction of a free-form answer into the schema.

    Without this, comparing constrained against free-form decoding measures the wrong
    thing. A free-form answer is typically correct *prose* -- "Shopping List: yoghurt,
    olive oil" -- which a JSON parser scores as a total failure. The first attempt at
    this check reported constrained decoding as +66.7pp better, which was entirely an
    artefact of the parser: parse rate was 0/12 because nothing had asked the model for
    JSON.

    Splitting reasoning from formatting is what makes the comparison fair: the tracking
    happens in the first call, the serialisation in the second. The extraction prompt
    forbids correcting anything, so the extractor cannot rescue a wrong answer.
    """
    if not raw.strip():
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(
        {"raw": raw, "model": model.name, "digest": model.digest,
         "tasks": [t.value for t in tasks], "mode": "extract"}, sort_keys=True
    ).encode()).hexdigest()
    cached = CACHE_DIR / f"{key}.json"
    if use_cache and cached.exists():
        return _parse(json.loads(cached.read_text())["content"])
    try:
        r = ollama.chat(model=model.name,
                        messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                                  {"role": "user", "content": raw}],
                        format=FinalState.schema_for(tasks),
                        options=model.options, think=model.think)
        content = r["message"]["content"]
    except Exception:
        return None
    if use_cache:
        cached.write_text(json.dumps({"content": content}))
    return _parse(content)


def verify_token_match(blocked: RunResult, interleaved: RunResult) -> tuple[bool, int]:
    """Confirm the two orderings really did tokenise to the same length.

    This is the answer to "how do you know the token match is real". The generator
    asserts the two conversations carry identical turn text; this checks what the model
    actually consumed, using Ollama's own `prompt_eval_count`. Reported rather than
    assumed, and any drifting pair is excluded from analysis.
    """
    delta = abs(blocked.prompt_tokens - interleaved.prompt_tokens)
    return delta == 0, delta


def resolve_model(name: str, **kw: Any) -> ModelSpec:
    """Look up a model's digest so each row records which weights actually answered.

    A tag like `gemma4:12b` can be repointed upstream at any time. Without the digest a
    silently swapped model would show up as a behavioural finding, which is the kind of
    result that is very hard to un-publish. Best-effort: if the lookup fails the run
    still proceeds, with an empty digest recorded so the gap is visible rather than
    implied.
    """
    digest = ""
    try:
        for m in ollama.list().get("models", []):
            if name in (m.get("model"), m.get("name")):
                digest = str(m.get("digest", ""))[:16]
                break
    except Exception:
        pass
    return ModelSpec(name=name, digest=digest, **kw)
