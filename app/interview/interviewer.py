"""Question generation: the LLM's second responsibility.

It writes the words of one question. It does not decide difficulty, topic or
whether the interview continues — those arrive already decided in the plan.
"""

from __future__ import annotations

from app.interview.grounding import Grounding
from app.interview.modes import InterviewMode
from app.interview.planner import PlannedAction
from app.interview.schemas import QuestionDraft
from app.interview.state import InterviewState
from app.llm.gateway import LLMGateway
from app.llm.schemas import LLMCallContext, LLMMessage

SYSTEM_PROMPT = """You are conducting a live technical interview for an AI/ML \
engineering role.

Interview: {display_name} — {focus}

Rules:
- Ask about exactly ONE thing. Never stack asks with "and", "additionally" or \
a list. If several angles tempt you, pick the single best one — follow-ups \
exist for the rest.
- Match length to the level. At difficulty 1-2, or when the candidate is \
struggling: one short, direct sentence. At difficulty 4-5 with a candidate \
doing well: one or two sentences of scenario or constraints may set up the \
question, but the ask itself stays a single sentence about that one thing.
- Interview, do not teach. Never reveal or hint at the answer.
- No praise, no encouragement, no commentary on the previous answer.
- Push on vagueness: ask why, ask for tradeoffs, ask for specifics.
- Stay professional and neutral, like a senior engineer who is short on time.
- Difficulty {difficulty} of 5, where 1 is junior and 5 is staff level.
- When the candidate's own resume claims are supplied, question those claims directly: ask what they decided, why, and what they would change. Never praise the claim and never assume it is true.
"""

_ACTION_INSTRUCTIONS: dict[str, str] = {
    "open_question": "Open the interview with a substantive question on the topic below.",
    "probe_deeper": "Probe deeper into the same thread. Target what the answer left unsaid.",
    "clarify": "The answer was vague or ambiguous. Ask for a precise clarification.",
    "challenge_assumption": "Challenge an assumption the candidate made, without correcting them.",
    "ask_tradeoff": "Ask them to justify a tradeoff or compare their choice against an alternative.",
    "increase_difficulty": "Raise the difficulty on this topic.",
    "decrease_difficulty": "Lower the difficulty and check the foundations of this topic.",
    "change_topic": "Move to the new topic below. Do not reference the previous topic.",
}


class Interviewer:
    def __init__(self, gateway: LLMGateway) -> None:
        self._gateway = gateway

    async def generate_question(
        self,
        *,
        plan: PlannedAction,
        state: InterviewState,
        mode: InterviewMode,
        grounding: Grounding | None = None,
        request_id: str | None = None,
    ) -> QuestionDraft:
        context = LLMCallContext(
            request_id=request_id,
            interview_id=state.interview_id,
            purpose="generate_question",
        )
        return await self._gateway.generate_structured(
            QuestionDraft,
            [
                LLMMessage(
                    role="user",
                    content=_user_prompt(plan, state, mode, grounding or Grounding()),
                )
            ],
            system=SYSTEM_PROMPT.format(
                display_name=mode.display_name,
                focus=mode.focus,
                difficulty=plan.difficulty,
            ),
            context=context,
        )


def _default_topic(mode: InterviewMode) -> str:
    return mode.topics[0] if mode.topics else mode.display_name


def _user_prompt(
    plan: PlannedAction,
    state: InterviewState,
    mode: InterviewMode,
    grounding: Grounding,
) -> str:
    instruction = _ACTION_INSTRUCTIONS.get(plan.action, _ACTION_INSTRUCTIONS["probe_deeper"])
    lines = [
        f"Action: {instruction}",
        f"Topic: {plan.topic or _default_topic(mode)}",
        f"Questions already asked: {len(state.questions_asked)}",
    ]

    target_skill = state.target_skill(plan.topic)
    if target_skill:
        lines.append(f"Skill the target role requires: {target_skill}")

    evidence = grounding.resume_evidence
    if evidence:
        lines += [
            "",
            "The candidate claims the following on their resume. Ground the "
            "question in these claims:",
            *(f"- {claim}" for claim in evidence),
        ]

    # On a topic change the previous exchange is deliberately withheld: telling
    # the model "do not reference the previous topic" while showing it the
    # previous answer loses to the model's urge to use what it can see (the
    # live smoke run demonstrated exactly that).
    if state.turns and plan.action != "change_topic":
        last = state.turns[-1]
        lines += [
            "",
            "Previous question:",
            last.question.text,
            "",
            "Candidate's answer:",
            last.answer.strip() or "(no answer given)",
        ]
        gaps = last.evaluation.missing_concepts + last.evaluation.misconceptions
        if gaps:
            lines += ["", "Gaps worth pressing on: " + ", ".join(gaps[:5])]

    if grounding.knowledge_notes:
        lines += [
            "",
            "Reference points a complete answer to this topic would cover. Use them "
            "to judge what to press on. Do NOT state them, hint at them, or read "
            "them out:",
            *(f"- {note}" for note in grounding.knowledge_notes),
        ]

    already_asked = [q.text for q in state.questions_asked[-3:]]
    if already_asked:
        lines += ["", "Do not repeat these questions:", *(f"- {q}" for q in already_asked)]

    return "\n".join(lines)
