"""Tests for the runner: message construction, cache keying, parsing, schema.

No model is called. Everything here is the deterministic scaffolding around inference,
which is exactly the part that silently corrupted results twice (D7 schema ambiguity,
D16 parser-vs-prose) and so deserves tests that do not need a GPU.
"""

import pytest

from taskswitch.generator import build_pair
from taskswitch.ops import TaskKind, assign_slots, slot_key
from taskswitch.runner import (
    ACK,
    FinalState,
    ModelSpec,
    RunResult,
    build_messages,
    cache_key,
    parse_slots,
    verify_token_match,
)

T2 = [TaskKind.SHOPPING, TaskKind.SCHEDULE]
T4 = [TaskKind.SHOPPING, TaskKind.SCHEDULE, TaskKind.SHOPPING, TaskKind.SCHEDULE]
M = ModelSpec(name="test", digest="abc")


# --- message construction ------------------------------------------------------------

def test_every_user_turn_is_followed_by_a_prefilled_ack():
    """The prefilled ack is what makes the token match provable: both orderings become
    the identical multiset of strings."""
    conv, _ = build_pair(1, T2, 8, 2, 2)
    msgs = build_messages(conv)
    assert msgs[0]["role"] == "system"
    body = msgs[1:-1]
    assert len(body) == 2 * len(conv.turns)
    for user, asst in zip(body[::2], body[1::2]):
        assert user["role"] == "user" and asst["role"] == "assistant"
        assert asst["content"] == ACK
    assert msgs[-1]["content"] == conv.final_request


@pytest.mark.parametrize("seed", range(15))
def test_paired_conversations_produce_identical_message_multisets(seed):
    """The token-match guarantee, checked at the message level."""
    b, i = build_pair(seed, T2, 8, 2, 6)
    assert sorted(m["content"] for m in build_messages(b)) == \
           sorted(m["content"] for m in build_messages(i))


# --- cache keying --------------------------------------------------------------------

def test_cache_key_is_stable_for_identical_input():
    conv, _ = build_pair(2, T2, 8, 2, 2)
    assert cache_key(conv, M, True) == cache_key(conv, M, True)


def test_cache_key_distinguishes_ordering_model_and_constraint():
    b, i = build_pair(2, T2, 8, 2, 2)
    keys = {cache_key(b, M, True), cache_key(i, M, True), cache_key(b, M, False),
            cache_key(b, ModelSpec(name="other", digest="z"), True)}
    assert len(keys) == 4


def test_cache_key_changes_when_the_digest_changes():
    """A tag can be repointed upstream. Keying on the digest means a swapped model
    invalidates the cache instead of silently reusing another model's answers."""
    conv, _ = build_pair(2, T2, 8, 2, 2)
    a = cache_key(conv, ModelSpec(name="m", digest="d1"), True)
    b = cache_key(conv, ModelSpec(name="m", digest="d2"), True)
    assert a != b


def test_cache_key_changes_with_generation_params():
    conv, _ = build_pair(2, T2, 8, 2, 2)
    a = cache_key(conv, ModelSpec(name="m", num_ctx=4096), True)
    b = cache_key(conv, ModelSpec(name="m", num_ctx=16384), True)
    assert a != b


# --- schema --------------------------------------------------------------------------

@pytest.mark.parametrize("tasks", [T2, T4, [TaskKind.SHOPPING], [TaskKind.SCHEDULE]])
def test_schema_has_one_field_per_slot(tasks):
    schema = FinalState.schema_for(tasks)
    props = schema["properties"]
    expected = [slot_key(t, s) for t, s in zip(tasks, assign_slots(tasks))]
    assert list(props) == expected
    assert schema["required"] == expected
    assert schema["additionalProperties"] is False


def test_schema_never_asks_for_an_untracked_slot():
    """An unrequested field could only ever be scored as a hallucination."""
    props = FinalState.schema_for([TaskKind.SHOPPING])["properties"]
    assert "schedule_0" not in props


def test_schedule_schema_uses_named_keys_not_positional_pairs():
    """D7 regression. A bare two-string array left the time/title order undefined, and
    the model resolved it DIFFERENTLY per condition -- manufacturing a switch cost."""
    item = FinalState.schema_for(T2)["properties"]["schedule_0"]["items"]
    assert item["type"] == "object"
    assert sorted(item["required"]) == ["time", "title"]
    assert item["additionalProperties"] is False


# --- parsing -------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    '{"shopping_0": ["milk"]}',
    '```json\n{"shopping_0": ["milk"]}\n```',
    'Here you go:\n{"shopping_0": ["milk"]}\nHope that helps.',
    '   {"shopping_0": ["milk"]}   ',
])
def test_parse_slots_tolerates_fenced_and_padded_json(raw):
    assert parse_slots(raw) == {"shopping_0": ["milk"]}


@pytest.mark.parametrize("raw", ["", "no json here", "**Shopping List:**\n- milk", "[1,2,3]"])
def test_parse_slots_returns_none_on_unparseable_input(raw):
    """Prose is exactly what free-form decoding returns -- correct, and unparseable.
    That is why the constrained-vs-free-form check needs an extraction pass (D16)."""
    assert parse_slots(raw) is None


def test_parse_slots_accepts_arbitrary_slot_names():
    """FinalState cannot express N slots, which is why the slot path uses a plain dict."""
    got = parse_slots('{"shopping_0":["a"],"shopping_1":["b"],"schedule_0":[]}')
    assert set(got) == {"shopping_0", "shopping_1", "schedule_0"}


def test_parse_slots_validates_the_requested_task_schema():
    raw = ('{"shopping_0":["milk"],'
           '"schedule_0":[{"time":"09:00","title":"standup"}]}')
    assert parse_slots(raw, T2) is not None


@pytest.mark.parametrize("raw", [
    '{"shopping_0":["milk"]}',
    ('{"shopping_0":["milk"],"schedule_0":[],'
     '"shopping_1":[]}'),
    '{"shopping_0":"milk","schedule_0":[]}',
    '{"shopping_0":[1],"schedule_0":[]}',
    '{"shopping_0":[],"schedule_0":[{"time":"09:00"}]}',
    ('{"shopping_0":[],"schedule_0":['
     '{"time":"09:00","title":"standup","room":"1A"}]}'),
    '{"shopping_0":[],"schedule_0":[{"time":900,"title":"standup"}]}',
])
def test_parse_slots_rejects_schema_violations(raw):
    assert parse_slots(raw, T2) is None


# --- token verification --------------------------------------------------------------

def _rr(tokens):
    conv, _ = build_pair(1, T2, 6, 0, 0)
    return RunResult(conversation=conv, model=M, raw_final="", parsed=None,
                     parse_ok=False, n_retries=0, wall_seconds=0.0, prompt_tokens=tokens)


def test_verify_token_match_is_exact_not_approximate():
    assert verify_token_match(_rr(500), _rr(500)) == (True, 0)
    assert verify_token_match(_rr(500), _rr(501)) == (False, 1)


def test_model_spec_options_carry_greedy_decoding():
    """Greedy decoding is why a parse-failure retry would be a no-op (D8)."""
    o = ModelSpec(name="m").options
    assert o["temperature"] == 0.0 and o["top_k"] == 1
