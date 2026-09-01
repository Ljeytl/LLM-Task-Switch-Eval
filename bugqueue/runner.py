"""Ollama runner for bugqueue conversations."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama

from bugqueue.generator import Conversation
from bugqueue.surface import output_schema

ACK = "Got it."
CACHE_DIR = Path("results/cache/bugs")


@dataclass
class ModelSpec:
    name: str
    digest: str = ""
    num_ctx: int = 16384
    temperature: float = 0.0
    top_k: int = 1
    seed: int = 0
    num_predict: int = 512
    think: bool = False

    @property
    def options(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_k": self.top_k, "seed": self.seed,
                "num_ctx": self.num_ctx, "num_predict": self.num_predict}


@dataclass
class RunResult:
    conversation: Conversation
    model: ModelSpec
    raw_final: str
    parsed: dict[str, Any] | None
    parse_ok: bool
    n_retries: int
    wall_seconds: float
    prompt_tokens: int = 0
    eval_tokens: int = 0
    constrained: bool = True
    error: str = ""


def build_messages(conv: Conversation) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": conv.system}]
    for turn in conv.turns:
        msgs.append({"role": "user", "content": turn})
        msgs.append({"role": "assistant", "content": ACK})
    msgs.append({"role": "user", "content": conv.final_request})
    return msgs


def cache_key(conv: Conversation, model: ModelSpec, constrained: bool) -> str:
    payload = json.dumps({
        "messages": build_messages(conv),
        "model": model.name, "digest": model.digest,
        "options": model.options, "think": model.think,
        "constrained": constrained, "domain": "bugqueue",
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse(raw: str) -> dict[str, Any] | None:
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


def run_conversation(conv: Conversation, model: ModelSpec, constrained: bool = True,
                     use_cache: bool = True) -> RunResult:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(conv, model, constrained)
    cached = CACHE_DIR / f"{key}.json"

    if use_cache and cached.exists():
        blob = json.loads(cached.read_text())
    else:
        messages = build_messages(conv)
        fmt = output_schema(conv.symbols) if constrained else None
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
            except Exception as exc:
                last_err, retries = f"{type(exc).__name__}: {exc}", attempt
                time.sleep(1.0 + attempt)
        if blob is None:
            blob = {"content": "", "prompt_eval_count": 0, "eval_count": 0,
                    "wall": 0.0, "retries": retries, "error": last_err}
        if use_cache and not blob["error"]:
            cached.write_text(json.dumps(blob))

    parsed = _parse(blob["content"])
    return RunResult(
        conversation=conv, model=model, raw_final=blob["content"], parsed=parsed,
        parse_ok=parsed is not None, n_retries=blob["retries"],
        wall_seconds=blob["wall"], prompt_tokens=blob["prompt_eval_count"],
        eval_tokens=blob["eval_count"], constrained=constrained, error=blob["error"],
    )


def resolve_model(name: str, **kw: Any) -> ModelSpec:
    digest = ""
    try:
        for m in ollama.list().get("models", []):
            if name in (m.get("model"), m.get("name")):
                digest = str(m.get("digest", ""))[:16]
                break
    except Exception:
        pass
    return ModelSpec(name=name, digest=digest, **kw)
