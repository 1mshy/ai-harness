"""Flashing safety. Pre-conditions before instructions, unconditional stop after.

AGENT_PLAN.md 9.2. ``review.safety_issue`` is true on 1,534 of the 30,584
reviewed calls (5.0%; the review block is present on only 79.4% of documents,
so every review-derived rate here understates by roughly a fifth). The plan's
breakdown of what those calls are: engine damage 301, drivability/limp mode
205, clutch/transmission 149, fuel/fire/EGT 123, turbo overpressure 108.

Re-measured here on 2026-08-06 against the same 1,534: ``technical_category``
is ``flashing_error`` on 666 and ``performance_issue`` on 503. That is the
whole point. **The safety surface and the number-one technical category are
the same population.** An agent walking someone through a flash is one wrong
file away from an immobilised or thermally unsafe car, and the corpus says so
in its own top symptom counts: limp mode 133, "won't start" / "car won't
start" / "no start" / "doesn't start" 158 combined, bricked 8, key not
detected 10.

Two mechanisms, in this order, and the order is not negotiable:

1. :func:`scan` -- an UNCONDITIONAL stop. If the customer has said the car
   will not start, is in limp mode, is glowing, is smoking, or has been
   towed, the turn ends and a human is paged. No retrieval, no diagnosis, no
   "have you tried". This runs before anything else and it is not gated on
   whether a flash is in progress, because by the time those words are said
   the flash already happened.

2. :func:`check_flash_preconditions` -- before *any* flashing instruction is
   emitted, three facts must be known: the file matches this ECU's hardware
   and software revision, the required hardware is present and current, and
   the fuel in the tank matches what the file expects. Unknown is not
   permission. Each unmet pre-condition carries the question to ask, so the
   agent's next turn is a question rather than an instruction.

:func:`guard_flash_instruction` composes the two and is what the agent loop
should call.

MEASURED, 2026-08-06, live Mongo. Reproduce with
``.venv/bin/python -m agentv1.guardrails.safety``:

    stop-scan over the 1,534 review.safety_issue calls   716  (46.7%)
    stop-scan over 1,512 review.safety_issue == false     36  ( 2.4%)

The 46.7% is deliberately NOT presented as recall to be optimised, and
pushing it higher would be the wrong move. ``review.safety_issue`` is the
reviewer's judgement over the whole call including the agent's half, the
outcome, and sometimes a prior contact; this gate fires only on physical
state the customer has stated in their own words. The 818 it does not fire on
are dominated by ``check engine light`` (40), ``misfiring`` (52 across three
spellings), ``EPC light`` (14) and ``won't flash`` (21+7) -- conditions where
the car is stationary and diagnosable and stopping the conversation would
help nobody. EPC was tried as a trigger and removed for that reason.

The 2.4% counter-rate is the cost, stated so it is visible. Over-stopping
buys a handoff; under-stopping buys the incident.

Nothing in this module is a diagnosis. It decides whether the agent is
allowed to keep talking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

Action = Literal["allow", "stop_and_escalate", "hold_for_preconditions"]


# ---------------------------------------------------------------------------
# 1. Unconditional stop
# ---------------------------------------------------------------------------
# Patterns are grouped by the physical state they describe, because the
# escalation record wants to say *what* happened, not just that something did.
# Every English pattern was drawn from the `symptoms` array of the 1,534
# safety_issue calls; the French forms come from the 94 French ones plus the
# Quebec shop vocabulary the ASR actually produces ("ça part pas", "il part
# pas", "remorquer").
_STOP_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "no_start",
        r"\b(?:wo?n'?t|will\s+not|does\s*n(?:'|o)?t|did\s*n(?:'|o)?t|would\s*n(?:'|o)?t"
        r"|cannot|can'?t|no|unable\s+to)\s+"
        r"(?:get\s+)?(?:start|starting|crank|turn\s+over|fire\s+up|key\s+(?:back\s+)?on"
        r"|ignition\s+power|go\s+into\s+(?:drive|gear|reverse))\b"
        r"|\bno[\s\-]start\b|\bnot\s+starting\b|\bkey\s+not\s+detected\b"
        r"|\bno\s+(?:ignition|crank)\b|\bd[eé]marre\s+(?:pas|plus)\b|\bne\s+d[eé]marre\b"
        r"|\bpart\s+(?:pas|plus)\b|\bveut\s+pas\s+partir\b"
        r"|\bne\s+veut\s+plus\s+d[eé]marrer\b",
        "The vehicle does not start.",
    ),
    (
        "bricked",
        r"\bbrick(?:ed|ing)?\b|\becu\s+is\s+(?:corrupt|dead|gone)\b"
        r"|\bcorrupt(?:ed)?\s+ecu\b|\bno\s+(?:comms?|communication)\s+with\s+the\s+ecu\b"
        r"|\becu\s+ne\s+r[eé]pond\s+plus\b|\bordinateur\s+est\s+mort\b",
        "The ECU is unresponsive or corrupt after a flash.",
    ),
    (
        "limp_mode",
        r"\blimp\s*(?:mode|home)?\b|\breduced\s+power\b|\bpull\s+over\s+now\b"
        r"|\bdrive\s+system\s+malfunction\b|\bsafe\s+mode\b|\bemergency\s+mode\b"
        r"|\b(?:flashing|blinking)\s+(?:check\s+)?engine\s+light\b"
        r"|\b(?:check\s+)?engine\s+light\s+(?:is\s+)?(?:flashing|blinking)\b"
        r"|\bmode\s+d[eé]grad[eé]\b|\bmode\s+de\s+secours\b|\bmode\s+s[eé]curis[eé]\b"
        r"|\bpuissance\s+r[eé]duite\b",
        "The vehicle is in limp mode or has lost drive.",
    ),
    (
        "thermal",
        r"\bglow(?:ing|ed)?\s*(?:red|orange|hot)?\b|\bred\s+hot\b|\bcherry\s+red\b"
        r"|\bmelt(?:ed|ing)?\b|\bscorch(?:ed|ing)?\b|\bburn(?:t|ed|ing)\b"
        r"|\boverheat\w*\b|\bsuper\s+hot\b|\bgetting\s+(?:really\s+|super\s+)?hot\b"
        r"|\begts?\b[^.\n]{0,30}\b(?:9[0-9]{2}|1[0-9]{3})\b"
        r"|\brougeoie\b|\bchauffe\s+au\s+rouge\b|\bsurchauff\w*\b|\bbr[uû]l[eé]\b"
        r"|\bfondu\b",
        "Something on the vehicle is glowing, melting or burnt.",
    ),
    (
        "smoke_fire",
        r"\bsmok(?:e|ing|ed)\b|\bsmoulder\w*\b|\bsmolder\w*\b|\bfire\b|\bflames?\b"
        r"|\bcaught\s+(?:on\s+)?fire\b|\bfum[eé]e\b|\ba\s+pris\s+(?:en\s+)?feu\b"
        r"|\bfeu\b|\bbr[uû]le\b",
        "Smoke or fire.",
    ),
    (
        "towed",
        r"\btow(?:ed|ing)\b|\bflat\s*bed(?:ded)?\b|\bhad\s+to\s+be\s+towed\b"
        r"|\bremorqu\w*\b|\bd[eé]panneuse\b",
        "The vehicle has been or must be towed.",
    ),
    (
        "stranded",
        r"\bstranded\b|\bstuck\s+on\s+the\s+(?:side\s+of\s+the\s+)?(?:road|highway)\b"
        r"|\bbroke\s+down\b|\bbroken\s+down\b|\bwon'?t\s+move\b|\bnon[\s\-]driv\w+\b"
        r"|\bstall(?:s|ed|ing)\b|\b(?:just\s+|immediately\s+)?died\s+(?:on\s+me|while|after)\b"
        r"|\bcuts?\s+out\s+(?:while|when|on)\b|\bshuts?\s+(?:it)?self\s+off\b"
        r"|\bimmobilis\w+\b|\bne\s+roule\s+plus\b|\ben\s+panne\b|\bcale\s+(?:tout\s+le\s+temps|souvent)\b",
        "The vehicle is immobilised away from a workshop.",
    ),
    (
        "mechanical_failure",
        r"\bblew\s+(?:up|a\s+\w+)\b|\bblown\s+(?:motor|engine|turbo|head\s*gasket|piston)\b"
        r"|\bgrenade[d]?\b|\bseized\b|\bspun\s+a\s+bearing\b|\bhydro\s*lock\w*\b"
        r"|\bcoolant\s+in\s+the\s+oil\b|\brod\s+(?:knock|through)\b"
        r"|\bmoteur\s+(?:est\s+)?(?:mort|fini|saut[eé])\b|\bcass[eé]\s+le\s+moteur\b",
        "A component has failed mechanically.",
    ),
)

# `fire` appears in "fire it up", `smoke` in "smoke test", `burn` in "burn a
# file to the cable", `limp` in nothing benign at all. These are the phrases
# that make an otherwise-alarming word innocent, and they are checked against
# the match window so one benign use does not suppress a real one elsewhere.
_STOP_EXEMPTIONS = (
    r"\bfire\s+it\s+up\b",
    r"\bfires?\s+(?:right\s+)?up\s+(?:fine|great|no\s+problem)\b",
    r"\bsmoke\s+test\b",
    r"\bburn(?:ing)?\s+(?:the\s+)?(?:file|map|software|image)\b",
    r"\bfirewall\b",
    r"\btowing\s+(?:capacity|package|mode)\b",
    r"\bcrank(?:ing)?\s+(?:sensor|position)\b",
)

_STOP_COMPILED = tuple(
    (rid, re.compile(pat, re.IGNORECASE), desc) for rid, pat, desc in _STOP_RULES
)
_EXEMPT_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in _STOP_EXEMPTIONS)


@dataclass(frozen=True)
class SafetyTrigger:
    rule_id: str
    description: str
    matched_text: str
    span: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "matched_text": self.matched_text,
            "span": list(self.span),
        }


@dataclass(frozen=True)
class Precondition:
    """One fact that must be known before a flashing instruction is emitted."""

    id: str
    satisfied: bool
    question: str
    why: str
    observed: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "satisfied": self.satisfied,
            "question": self.question,
            "why": self.why,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class SafetyVerdict:
    action: Action
    triggers: tuple[SafetyTrigger, ...] = ()
    preconditions: tuple[Precondition, ...] = ()
    message: str | None = None
    escalate: bool = False
    language: str = "en"

    @property
    def blocked(self) -> bool:
        return self.action != "allow"

    @property
    def unmet(self) -> tuple[Precondition, ...]:
        return tuple(p for p in self.preconditions if not p.satisfied)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "triggers": [t.as_dict() for t in self.triggers],
            "preconditions": [p.as_dict() for p in self.preconditions],
            "unmet": [p.id for p in self.unmet],
            "message": self.message,
            "escalate": self.escalate,
            "language": self.language,
        }


_STOP_MESSAGE = {
    "en": (
        "I'm stopping here. What you're describing is a vehicle that isn't "
        "safe to work on over chat, and giving you another step to try could "
        "make it worse or put you in a dangerous position. I'm handing this "
        "to a technician now -- please don't attempt another flash, and don't "
        "drive the vehicle."
    ),
    "fr": (
        "J'arrete ici. Ce que vous decrivez est un vehicule qu'il n'est pas "
        "prudent de depanner par clavardage, et vous donner une autre etape a "
        "essayer risquerait d'aggraver la situation ou de vous mettre en "
        "danger. Je transfere a un technicien tout de suite -- n'essayez pas "
        "une autre programmation et ne conduisez pas le vehicule."
    ),
}


def _lang(language: str | None) -> str:
    return "fr" if (language or "en").lower().startswith("fr") else "en"


def _safe_excerpt(matched: str) -> str:
    """Redact a trigger excerpt before it is ever stored on the verdict.

    ``matched_text`` is customer text. Most rules match a fixed phrase, but the
    ``thermal`` rule's EGT branch (``\\begts?\\b[^.\\n]{0,30}\\b(9\\d\\d|1\\d{3})\\b``)
    deliberately spans up to 30 characters of whatever the caller typed in
    between, so an arbitrary substring of the inbound turn can end up here. It
    then travels: ``as_dict()`` copies it into the escalation/log record and
    ``compose.py`` serialises ``str(trigger)`` into ``Verdict.detail``. Measured
    leak before this call existed -- "EGT call me back at 514 555 1234" put the
    caller's phone number into the detail dict of a *safety* verdict.

    CONTRACT.md non-negotiable 2 is about every egress, not just the payload
    allowlist, so the excerpt is scrubbed at construction rather than at each
    render site: a later ``as_dict()`` or f-string cannot reintroduce what the
    object never held. Redaction failing must never suppress a safety stop, so
    the fallback drops the excerpt entirely instead of raising.
    """
    if not matched:
        return matched
    try:
        from . import pii

        return pii.redact(matched)
    except Exception:  # noqa: BLE001 -- a safety stop outranks its own excerpt
        return "[REDACTION_UNAVAILABLE]"


def scan(text: str, *, language: str | None = None) -> SafetyVerdict:
    """UNCONDITIONAL stop-and-escalate check. Call on every inbound turn.

    Not gated on whether the conversation is about flashing. By the time
    somebody types "it won't start", the flash is already in the past tense,
    and a gate that only ran during a flashing dialogue would have been
    skipped by the customer who opened with the failure.
    """
    lang = _lang(language)
    if not text or not text.strip():
        return SafetyVerdict(action="allow", language=lang)

    exempt = [m.span() for rx in _EXEMPT_COMPILED for m in rx.finditer(text)]
    triggers: list[SafetyTrigger] = []
    for rid, rx, desc in _STOP_COMPILED:
        for m in rx.finditer(text):
            lo, hi = m.span()
            if any(not (ehi <= lo or elo >= hi) for elo, ehi in exempt):
                continue
            triggers.append(
                SafetyTrigger(
                    rule_id=rid,
                    description=desc,
                    matched_text=_safe_excerpt(m.group(0)),
                    span=(lo, hi),
                )
            )
            break  # one trigger per rule is enough; the record wants kinds, not counts

    if not triggers:
        return SafetyVerdict(action="allow", language=lang)
    return SafetyVerdict(
        action="stop_and_escalate",
        triggers=tuple(triggers),
        message=_STOP_MESSAGE[lang],
        escalate=True,
        language=lang,
    )


# ---------------------------------------------------------------------------
# 2. Pre-conditions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlashContext:
    """Everything that must be known before a flashing instruction is emitted.

    Every field defaults to None, and None means *unknown*, which is treated
    exactly like *wrong*. That asymmetry is the whole design: the failure mode
    in the corpus is not an agent who checked and got it wrong, it is an agent
    who never checked. There is no ``force`` flag and no ``skip_checks``
    argument -- a caller who wants to bypass this has to not call it, which is
    visible in a diff.
    """

    # ECU identity as read off the vehicle, and as the candidate file declares
    # it. Both are needed: a match on box code alone is not a match, because
    # the same box code ships with several software revisions and the software
    # revision is what determines whether the calibration lands.
    vehicle_ecu_box_code: str | None = None
    vehicle_ecu_software_number: str | None = None
    file_ecu_box_codes: tuple[str, ...] = ()
    file_ecu_software_numbers: tuple[str, ...] = ()

    # Hardware. `cable_model` is the UniConnect/UniCONNECT+ tier; the two are
    # different products (see text/normalize.py on why the plus sign matters)
    # and a file that needs the plus cable will not push over the base one.
    cable_model: str | None = None
    cable_firmware_current: bool | None = None
    required_cable_model: str | None = None
    required_hardware: tuple[str, ...] = ()
    installed_hardware: tuple[str, ...] = ()

    # Fuel. Flashing a 93-octane file into a car that will next be filled with
    # 87 is the fuel/fire/EGT bucket -- 123 of the 1,534.
    file_fuel_octane: str | None = None
    vehicle_fuel_octane: str | None = None

    # Power state. A flash that loses power mid-write is the bricking path.
    battery_maintainer_connected: bool | None = None

    # Free-form: what the customer just said, so `scan` can run on the same
    # object and the caller cannot forget to.
    customer_text: str = ""
    language: str = "en"


def _norm_code(code: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


_QUESTIONS = {
    "en": {
        "ecu_revision": (
            "Before anything else -- can you read the ECU's box code and "
            "software number off the vehicle with the cable connected, and "
            "send me both exactly as they appear?"
        ),
        "hardware": (
            "Which cable are you using, and is its firmware up to date? Also "
            "let me know which supporting hardware is actually fitted to the "
            "car right now."
        ),
        "fuel": (
            "What fuel is in the tank right now, and what will you be running "
            "day to day? I need the octane before I can point you at a file."
        ),
        "power": (
            "Is the car on a battery maintainer or charger? A flash that loses "
            "power part-way through is the one failure we cannot undo remotely."
        ),
    },
    "fr": {
        "ecu_revision": (
            "Avant toute chose -- pouvez-vous lire le box code et le numero de "
            "logiciel de l'ECU sur le vehicule, cable branche, et me les "
            "envoyer exactement tels qu'affiches?"
        ),
        "hardware": (
            "Quel cable utilisez-vous, et son micrologiciel est-il a jour? "
            "Dites-moi aussi quel materiel est reellement installe sur "
            "l'auto en ce moment."
        ),
        "fuel": (
            "Quel carburant est dans le reservoir presentement, et lequel "
            "utiliserez-vous au quotidien? J'ai besoin de l'indice d'octane "
            "avant de vous diriger vers un fichier."
        ),
        "power": (
            "L'auto est-elle branchee sur un chargeur ou un mainteneur de "
            "batterie? Une programmation qui manque de courant en cours de "
            "route est la seule panne qu'on ne peut pas defaire a distance."
        ),
    },
}

_WHY = {
    "ecu_revision": (
        "Wrong file for the ECU revision is the dominant failure in the "
        "corpus: technical_category is flashing_error on 666 of the 1,534 "
        "calls carrying review.safety_issue."
    ),
    "hardware": (
        "A file that assumes hardware the car does not have runs the engine "
        "outside its calibration; turbo overpressure accounts for 108 of the "
        "1,534."
    ),
    "fuel": (
        "Octane mismatch is the fuel/fire/EGT bucket -- 123 of the 1,534 -- "
        "and it does not fail at flash time, it fails on the next tank."
    ),
    "power": (
        "An interrupted write is how a vehicle arrives at 'bricked/"
        "unresponsive after an interrupted flash back to stock', which is a "
        "verbatim safety_detail from the corpus."
    ),
}


def check_flash_preconditions(ctx: FlashContext) -> SafetyVerdict:
    """Gate on the three facts, plus power state. Unknown counts as unmet."""
    lang = _lang(ctx.language)
    q = _QUESTIONS[lang]
    checks: list[Precondition] = []

    # --- correct file for this ECU revision ---------------------------------
    box_ok = bool(ctx.vehicle_ecu_box_code) and _norm_code(
        ctx.vehicle_ecu_box_code
    ) in {_norm_code(c) for c in ctx.file_ecu_box_codes}
    sw_ok = bool(ctx.vehicle_ecu_software_number) and _norm_code(
        ctx.vehicle_ecu_software_number
    ) in {_norm_code(c) for c in ctx.file_ecu_software_numbers}
    checks.append(
        Precondition(
            id="ecu_revision",
            satisfied=box_ok and sw_ok,
            question=q["ecu_revision"],
            why=_WHY["ecu_revision"],
            observed=(
                f"box={ctx.vehicle_ecu_box_code!r} in_file_list={box_ok}; "
                f"sw={ctx.vehicle_ecu_software_number!r} in_file_list={sw_ok}"
            ),
        )
    )

    # --- required hardware present -----------------------------------------
    cable_ok = (
        ctx.cable_model is not None
        and ctx.cable_firmware_current is True
        and (
            ctx.required_cable_model is None
            or _norm_code(ctx.cable_model) == _norm_code(ctx.required_cable_model)
        )
    )
    installed = {_norm_code(h) for h in ctx.installed_hardware}
    hw_missing = [h for h in ctx.required_hardware if _norm_code(h) not in installed]
    checks.append(
        Precondition(
            id="hardware",
            satisfied=cable_ok and not hw_missing,
            question=q["hardware"],
            why=_WHY["hardware"],
            observed=(
                f"cable={ctx.cable_model!r} firmware_current="
                f"{ctx.cable_firmware_current!r} required={ctx.required_cable_model!r}; "
                f"missing_hardware={hw_missing}"
            ),
        )
    )

    # --- fuel matches the file ---------------------------------------------
    fuel_ok = (
        ctx.file_fuel_octane is not None
        and ctx.vehicle_fuel_octane is not None
        and _norm_code(ctx.file_fuel_octane) == _norm_code(ctx.vehicle_fuel_octane)
    )
    checks.append(
        Precondition(
            id="fuel",
            satisfied=fuel_ok,
            question=q["fuel"],
            why=_WHY["fuel"],
            observed=(
                f"file={ctx.file_fuel_octane!r} vehicle={ctx.vehicle_fuel_octane!r}"
            ),
        )
    )

    # --- stable power -------------------------------------------------------
    checks.append(
        Precondition(
            id="power",
            satisfied=ctx.battery_maintainer_connected is True,
            question=q["power"],
            why=_WHY["power"],
            observed=f"battery_maintainer={ctx.battery_maintainer_connected!r}",
        )
    )

    unmet = [c for c in checks if not c.satisfied]
    if not unmet:
        return SafetyVerdict(
            action="allow", preconditions=tuple(checks), language=lang
        )
    return SafetyVerdict(
        action="hold_for_preconditions",
        preconditions=tuple(checks),
        # The agent's next turn is the first unmet question, not a list of
        # four. Asking four at once is how a customer answers none of them.
        message=unmet[0].question,
        escalate=False,
        language=lang,
    )


def guard_flash_instruction(ctx: FlashContext) -> SafetyVerdict:
    """The single call the agent loop makes before emitting flashing steps.

    Stop check first. A car that is already dead does not need its
    pre-conditions collected; it needs a person.
    """
    stop = scan(ctx.customer_text, language=ctx.language)
    if stop.blocked:
        return stop
    return check_flash_preconditions(ctx)


# ---------------------------------------------------------------------------
# Self-check:  .venv/bin/python -m agentv1.guardrails.safety
# ---------------------------------------------------------------------------
def _customer_text(doc: dict[str, Any]) -> str:
    return " ".join(
        str(t.get("text") or "")
        for t in (doc.get("conversation_turns") or [])
        if t.get("speaker_role") == "CUSTOMER"
    )


def _self_check() -> int:
    from ..clients.mongo import source_db

    fails = 0

    # --- fixed strings ------------------------------------------------------
    must_stop = [
        ("the car won't start after the flash", "en"),
        ("it's in limp mode and the engine light is flashing", "en"),
        ("the turbo was glowing red on the dyno", "en"),
        ("there's smoke coming from under the hood", "en"),
        ("I had it towed to the shop this morning", "en"),
        ("the ECU is corrupt, it's bricked", "en"),
        ("mon auto ne demarre plus depuis le flash", "fr"),
        ("il est en mode degrade", "fr"),
        ("j'ai du le faire remorquer", "fr"),
    ]
    must_pass = [
        ("what stage should I run on 93 octane", "en"),
        ("how long does the flash usually take", "en"),
        ("it fires right up, no problem, just want more power", "en"),
        ("do you do a smoke test for boost leaks", "en"),
        ("what's the towing capacity of the Q7", "en"),
        ("quel est le prix du stage 2", "fr"),
    ]
    for txt, lang in must_stop:
        if not scan(txt, language=lang).blocked:
            fails += 1
            print(f"  FAIL should stop: {txt!r}")
    for txt, lang in must_pass:
        v = scan(txt, language=lang)
        if v.blocked:
            fails += 1
            print(f"  FAIL should pass: {txt!r} -> {[t.rule_id for t in v.triggers]}")
    print(f"stop-scan fixed cases: {len(must_stop) + len(must_pass) - fails}"
          f"/{len(must_stop) + len(must_pass)} correct")

    # --- regression: a trigger excerpt must not carry PII --------------------
    # The `thermal` rule's EGT branch spans up to 30 characters of arbitrary
    # caller text. Before `_safe_excerpt`, this exact string put the caller's
    # phone number into SafetyTrigger.matched_text, and from there into
    # as_dict(), into the escalation record, and into compose.py's
    # `Verdict.detail` by way of str(trigger). CONTRACT.md non-negotiable 2.
    leaky = "EGT call me back at 514 555 1234"
    lv = scan(leaky)
    rendered = str(lv.as_dict()) + " " + " ".join(str(t) for t in lv.triggers)
    if not lv.blocked:
        fails += 1
        print("  FAIL leak-probe string no longer trips the thermal rule; "
              "the regression case has stopped testing anything")
    elif "555 1234" in rendered or "514 555" in rendered:
        fails += 1
        print(f"  FAIL trigger excerpt leaked a phone number: {rendered}")
    else:
        print(f"trigger excerpt redacted -> {lv.triggers[0].matched_text!r}")

    # --- pre-conditions -----------------------------------------------------
    empty = check_flash_preconditions(FlashContext())
    if empty.action != "hold_for_preconditions" or len(empty.unmet) != 4:
        fails += 1
        print(f"  FAIL empty context should hold on all 4, got {empty.as_dict()}")
    else:
        print(f"empty context -> {empty.action}, unmet {[p.id for p in empty.unmet]}")

    good = FlashContext(
        vehicle_ecu_box_code="8V0 906 259 K",
        vehicle_ecu_software_number="0001",
        file_ecu_box_codes=("8V0906259K",),
        file_ecu_software_numbers=("0001",),
        cable_model="UniCONNECT+",
        required_cable_model="uniconnect+",
        cable_firmware_current=True,
        required_hardware=("downpipe",),
        installed_hardware=("Downpipe",),
        file_fuel_octane="93",
        vehicle_fuel_octane="93",
        battery_maintainer_connected=True,
    )
    v = check_flash_preconditions(good)
    if v.action != "allow":
        fails += 1
        print(f"  FAIL fully-specified context should allow, got {v.as_dict()}")
    else:
        print("fully-specified context -> allow")

    wrong_fuel = FlashContext(**{**good.__dict__, "vehicle_fuel_octane": "87"})
    v = check_flash_preconditions(wrong_fuel)
    if v.action != "hold_for_preconditions" or [p.id for p in v.unmet] != ["fuel"]:
        fails += 1
        print(f"  FAIL octane mismatch should hold on fuel only, got {v.as_dict()}")
    else:
        print("93-octane file into an 87-octane tank -> hold_for_preconditions [fuel]")

    # A dead car short-circuits the pre-conditions entirely.
    dead = FlashContext(**{**good.__dict__, "customer_text": "it won't start now"})
    v = guard_flash_instruction(dead)
    if v.action != "stop_and_escalate":
        fails += 1
        print(f"  FAIL dead car should stop even with preconditions met, got {v.action}")
    else:
        print("all pre-conditions met but 'it won't start' -> stop_and_escalate")

    # --- coverage over the real safety population ---------------------------
    db = source_db()
    coll = db["calls_analysis"]
    unsafe = list(
        coll.find(
            {"review.safety_issue": True},
            {"conversation_turns": 1, "language": 1, "symptoms": 1},
        )
    )
    hit = sum(
        1
        for d in unsafe
        if scan(
            _customer_text(d) + " " + " ".join(str(s) for s in (d.get("symptoms") or [])),
            language=d.get("language") or "en",
        ).blocked
    )
    print(f"\nstop-scan over the {len(unsafe)} review.safety_issue calls")
    print(f"  would have stopped  {hit}  ({100.0 * hit / len(unsafe):.1f}%)")
    print("  NOT a recall figure to optimise: review.safety_issue includes")
    print("  outcomes the agent could not have seen in the customer's words")
    print("  (an agent-side observation, a follow-up call, an RMA), and this")
    print("  gate deliberately fires only on stated physical state.")

    control = list(
        coll.find(
            {
                "review.safety_issue": False,
                "call_id": {"$mod": [19, 3]},
            },
            {"conversation_turns": 1, "language": 1},
        )
    )
    cfp = sum(
        1
        for d in control
        if scan(_customer_text(d), language=d.get("language") or "en").blocked
    )
    print(f"\nstop-scan over {len(control)} calls with review.safety_issue false")
    print(f"  would have stopped  {cfp}  ({100.0 * cfp / max(len(control), 1):.1f}%)")
    print("  Over-stopping here costs a handoff, not an incident. The rate is")
    print("  reported so the cost is visible, not so it can be minimised.")

    print(f"\nself-check {'PASS' if fails == 0 else 'FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
