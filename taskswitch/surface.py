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

from .ops import Op, OpKind, TaskKind, slot_label

#: Per-instance vocabularies. Each task slot draws from a DISJOINT pool, which is what
#: makes misattribution unmistakable: "screws" appearing on the grocery list is
#: unambiguous in a way that a second generic list would never be. The original
#: two-task design got this property from using two different task *kinds*; naming the
#: instances preserves it while lifting the two-task cap.
SLOT_ITEMS: tuple[tuple[str, ...], ...] = (
    ("milk", "eggs", "bread", "rice", "apples", "coffee", "butter", "spinach",
     "yoghurt", "olive oil", "tomatoes", "cheese", "onions", "pasta", "lemons"),
    ("screws", "wood glue", "sandpaper", "masking tape", "hinges", "wall plugs",
     "paint brush", "spirit level", "drill bits", "picture hooks"),
    ("plasters", "ibuprofen", "throat lozenges", "antiseptic", "vitamin d",
     "eye drops", "bandages", "sun cream"),
    ("compost", "tomato seeds", "plant pots", "twine", "secateurs", "bulbs"),
)

SLOT_TITLES: tuple[tuple[str, ...], ...] = (
    ("standup", "design review", "one-on-one", "retro", "planning", "demo",
     "budget sync", "onboarding", "postmortem", "roadmap review"),
    ("dentist", "haircut", "gym session", "physio", "book club", "piano lesson",
     "car service", "eye test"),
    ("sprint review", "guild sync", "hiring panel", "all hands", "office hours"),
    ("school run", "swimming", "grandma visit", "vet appointment", "birthday party"),
)

#: Backwards-compatible aliases for slot 0.
ITEMS: tuple[str, ...] = SLOT_ITEMS[0]
TITLES: tuple[str, ...] = SLOT_TITLES[0]


def vocabulary(task: TaskKind, slot: int) -> tuple[str, ...]:
    """The disjoint entity pool belonging to one task instance."""
    pools = SLOT_ITEMS if task is TaskKind.SHOPPING else SLOT_TITLES
    return pools[slot % len(pools)]

_TEMPLATES: dict[tuple[TaskKind, OpKind], tuple[str, ...]] = {
    (TaskKind.SHOPPING, OpKind.ADD): (
        "Please add {item} to my {slot}.",
        "Can you put {item} on the {slot}?",
        "I need to pick up {item} — add it to the {slot}.",
        "Add {item} to the {slot}, please.",
        "Stick {item} on the {slot} for me.",
    ),
    (TaskKind.SHOPPING, OpKind.REMOVE): (
        "Take {item} off the {slot}.",
        "Remove {item} from the {slot}, I already have some.",
        "Scratch {item} from the {slot}.",
        "Drop {item} from my {slot} please.",
        "I don't need {item} any more — take it off the {slot}.",
    ),
    (TaskKind.SHOPPING, OpKind.QUERY): (
        "Is {item} on the {slot} right now?",
        "Have I got {item} on the {slot}?",
        "Quick check — does the {slot} include {item}?",
        "Am I already buying {item} on the {slot}?",
    ),
    (TaskKind.SHOPPING, OpKind.FALSE_ASSERT): (
        "I thought about adding {item} to the {slot}, but decided against it.",
        "Someone mentioned {item} for the {slot}, though we're not buying it.",
        "I nearly put {item} on the {slot} and then changed my mind.",
        "{item} was suggested for the {slot} — I'm not adding it though.",
    ),
    (TaskKind.SHOPPING, OpKind.NOISE): (
        "Prices have really gone up this year.",
        "The shop near me reorganised all its aisles last week.",
        "I keep forgetting my reusable bags when I go out.",
        "There was a huge queue at the checkout yesterday.",
    ),
    (TaskKind.SCHEDULE, OpKind.ADD): (
        "Schedule {title} at {time} on my {slot}.",
        "Put {title} in my {slot} for {time}.",
        "Book {title} for {time} on the {slot} please.",
        "Add a {time} slot for {title} to the {slot}.",
        "I need {title} on the {slot} at {time}.",
    ),
    (TaskKind.SCHEDULE, OpKind.UPDATE): (
        "Move {title} to {new_time} on my {slot}.",
        "Can you shift {title} to {new_time} in the {slot}?",
        "Reschedule {title} for {new_time} on the {slot}.",
        "{title} on the {slot} needs to move to {new_time}.",
        "Push {title} back to {new_time} in my {slot}.",
    ),
    (TaskKind.SCHEDULE, OpKind.REMOVE): (
        "Cancel {title} on my {slot}.",
        "Take {title} off my {slot}.",
        "Drop {title} from the {slot}.",
        "{title} is cancelled — remove it from the {slot}.",
    ),
    (TaskKind.SCHEDULE, OpKind.QUERY): (
        "What time is {title} on my {slot}?",
        "When is {title} scheduled for on the {slot}?",
        "Remind me when {title} is on the {slot}.",
        "Is {title} still on the {slot}?",
    ),
    (TaskKind.SCHEDULE, OpKind.FALSE_ASSERT): (
        "I considered booking {title} at {time} on the {slot}, but I won't.",
        "There was talk of {title} at {time} for the {slot} — it never happened.",
        "I almost put {title} at {time} in the {slot} and then didn't.",
        "Someone floated {title} for {time} on the {slot}; we decided against it.",
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

    Every template names the specific list or calendar, because with more than one
    instance of a kind the turn would otherwise be ambiguous -- and an ambiguous turn
    would make the answer key unanswerable rather than merely hard.

    Template choice is drawn from `rng`, so a conversation is fully reproducible from
    its seed while no single phrasing dominates.
    """
    templates = _TEMPLATES[(op.task, op.kind)]
    return rng.choice(templates).format(slot=slot_label(op.task, op.slot), **op.payload)


def render_final_request(tasks: list[TaskKind]) -> str:
    """The last user turn: asks for the current state of every tracked task.

    Deliberately does not restate what the tasks contain or hint at their size, since
    either would leak part of the answer.
    """
    from .ops import assign_slots
    labels = [slot_label(t, s) for t, s in zip(tasks, assign_slots(tasks))]
    listed = ", ".join(f"my {n}" for n in labels[:-1])
    listed = f"{listed} and my {labels[-1]}" if len(labels) > 1 else f"my {labels[0]}"
    return (f"Now give me the current state of {listed}. "
            f"Report exactly what is on each one right now.")


def system_prompt(tasks: list[TaskKind]) -> str:
    """Names the tasks being tracked but never their contents.

    Held constant across orderings, so it cannot contribute to a token-count
    difference between a blocked and an interleaved conversation.
    """
    from .ops import assign_slots
    labels = [slot_label(t, s) for t, s in zip(tasks, assign_slots(tasks))]
    articled = [f"a {n}" for n in labels]
    listed = (", ".join(articled[:-1]) + f" and {articled[-1]}") if len(articled) > 1 \
        else articled[0]
    return (f"You are an assistant helping the user manage {listed} over the course "
            f"of a conversation. These are separate lists -- keep them apart. Track "
            f"every change the user makes and report the current state accurately when "
            f"asked.")
