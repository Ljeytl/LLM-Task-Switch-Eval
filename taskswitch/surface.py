"""Natural-language surface forms for operations.

Every user turn is templated, never model-generated. A model-generated turn could
say something the op list does not encode, which would break the exactness of the
ground truth -- and exact ground truth is the whole reason this project needs no
judge. The cost is that turns read less naturally than real users, which is stated
as a limitation rather than papered over.

At least four paraphrases exist per (task, kind) and the choice is drawn from a
seeded RNG, so a model cannot pass by pattern-matching one fixed phrasing while
failing to track state.

Two op kinds need care because their wording decides whether the probe is fair:

- FALSE_ASSERT must be unambiguously *counterfactual*. "I put rice on the list" is a
  legitimate instruction in ordinary conversation, so scoring a model wrong for
  honouring it would be scoring our own phrasing. These templates negate or
  hypothesise instead ("I thought about adding rice, but decided not to"), which
  leaves exactly one correct behaviour: do not change state.
- NOISE is topical rather than generic. An aside that mentions groceries is a harder
  distractor than one about the weather, and NOISE doubles as the lever that varies
  context length independently of state load.
"""

from __future__ import annotations

from random import Random

from .ops import Op, OpKind, TaskKind

ITEMS: tuple[str, ...] = (
    "milk", "eggs", "bread", "rice", "apples", "coffee", "butter", "spinach",
    "yoghurt", "olive oil", "tomatoes", "cheese", "onions", "pasta", "lemons",
)

TITLES: tuple[str, ...] = (
    "standup", "design review", "one-on-one", "retro", "planning", "demo",
    "budget sync", "onboarding", "postmortem", "roadmap review",
)

_TEMPLATES: dict[tuple[TaskKind, OpKind], tuple[str, ...]] = {
    (TaskKind.SHOPPING, OpKind.ADD): (
        "Please add {item} to my shopping list.",
        "Can you put {item} on the list?",
        "I need to pick up {item} — add it.",
        "Add {item} to the groceries, please.",
        "Stick {item} on the shopping list for me.",
    ),
    (TaskKind.SHOPPING, OpKind.REMOVE): (
        "Take {item} off the shopping list.",
        "Remove {item} from the list, I already have some.",
        "Scratch {item} from the groceries.",
        "Drop {item} from my shopping list please.",
        "I don't need {item} any more — take it off.",
    ),
    (TaskKind.SHOPPING, OpKind.QUERY): (
        "Is {item} on the shopping list right now?",
        "Have I got {item} on the list?",
        "Quick check — does the list include {item}?",
        "Am I already buying {item}?",
    ),
    (TaskKind.SHOPPING, OpKind.FALSE_ASSERT): (
        "I thought about adding {item}, but decided against it.",
        "My partner mentioned {item}, though we're not buying it this week.",
        "I nearly put {item} on the list and then changed my mind.",
        "Someone suggested {item} — I'm not adding it though.",
    ),
    (TaskKind.SHOPPING, OpKind.NOISE): (
        "Grocery prices have really gone up this year.",
        "The shop near me reorganised all its aisles last week.",
        "I keep forgetting my reusable bags when I go shopping.",
        "There was a huge queue at the checkout yesterday.",
    ),
    (TaskKind.SCHEDULE, OpKind.ADD): (
        "Schedule {title} at {time}.",
        "Put {title} in my calendar for {time}.",
        "Book {title} for {time} please.",
        "Add a {time} slot for {title}.",
        "I need {title} on the schedule at {time}.",
    ),
    (TaskKind.SCHEDULE, OpKind.UPDATE): (
        "Move {title} to {new_time}.",
        "Can you shift {title} to {new_time}?",
        "Reschedule {title} for {new_time}.",
        "{title} needs to move to {new_time}.",
        "Push {title} back to {new_time} please.",
    ),
    (TaskKind.SCHEDULE, OpKind.REMOVE): (
        "Cancel {title}.",
        "Take {title} off my calendar.",
        "Drop {title} from the schedule.",
        "{title} is cancelled — remove it.",
    ),
    (TaskKind.SCHEDULE, OpKind.QUERY): (
        "What time is {title}?",
        "When is {title} scheduled for?",
        "Remind me when {title} is.",
        "Is {title} still on the calendar?",
    ),
    (TaskKind.SCHEDULE, OpKind.FALSE_ASSERT): (
        "I considered booking {title} at {time}, but I won't.",
        "There was talk of a {title} at {time} — it never got scheduled.",
        "I almost put {title} at {time} in the calendar and then didn't.",
        "Someone floated {title} for {time}; we decided against it.",
    ),
    (TaskKind.SCHEDULE, OpKind.NOISE): (
        "My calendar app keeps sending me duplicate notifications.",
        "Meetings always seem to cluster on the same day.",
        "I switched to a new timezone setting last month.",
        "The conference room booking system was down yesterday.",
    ),
}


def render(op: Op, rng: Random) -> str:
    """One natural-language user turn for this op.

    Template choice is drawn from `rng`, so a conversation is fully reproducible from
    its seed while no single phrasing dominates.
    """
    templates = _TEMPLATES[(op.task, op.kind)]
    return rng.choice(templates).format(**op.payload)


def render_final_request(tasks: list[TaskKind]) -> str:
    """The last user turn: asks for the current state of every tracked task.

    Deliberately does not restate what the tasks contain or hint at their size, since
    either would leak part of the answer.
    """
    names = {TaskKind.SHOPPING: "shopping list", TaskKind.SCHEDULE: "meeting schedule"}
    listed = " and my ".join(names[t] for t in tasks)
    return (f"Now give me the current state of my {listed}. "
            f"Report exactly what is on each one right now.")


def system_prompt(tasks: list[TaskKind]) -> str:
    """Names the tasks being tracked but never their contents.

    Held constant across orderings, so it cannot contribute to a token-count
    difference between a blocked and an interleaved conversation.
    """
    names = {TaskKind.SHOPPING: "a shopping list", TaskKind.SCHEDULE: "a meeting schedule"}
    listed = " and ".join(names[t] for t in tasks)
    return (f"You are an assistant helping the user manage {listed} over the course of "
            f"a conversation. Track every change the user makes and report the current "
            f"state accurately when asked.")
