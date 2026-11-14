"""LLM email generator. Writes from the same EmailContext as the template, and is told to use
nothing else, so claims stay grounded in retrieved data."""
import json

from app.schemas import EmailContext, StudentProfile
from app.services.email.context import ProfessorInfo
from app.services.llm_client import LLMUnavailable, complete_json

__all__ = ["generate", "LLMUnavailable", "PROMPT_VERSION"]

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You write short cold emails from students to professors asking about research positions.

Rules:
- Use only facts present in the <student>, <professor> and <grounding> blocks. Never invent
  publications, results, courses, skills, or experiences.
- If <grounding> names a paper, mention it by title once. If not, do not mention any paper.
- 150-220 words in the body. Plain, specific, professional; no flattery, no buzzwords.
- Address the professor as "Professor <Last name>". Sign with the student's name and email.
- Mention that a resume is attached.
"""

_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"],
    "additionalProperties": False,
}

def generate(prof: ProfessorInfo, student: StudentProfile, ctx: EmailContext,
             paper_snippet: str | None = None) -> tuple[str, str]:
    grounding = ctx.model_dump()
    if paper_snippet:
        grounding["paper_abstract_excerpt"] = paper_snippet
    user = (
        f"<student>\n{student.model_dump_json(indent=1)}\n</student>\n"
        f"<professor>\n{json.dumps({'name': prof.name, 'research_summary': prof.summary}, indent=1)}\n</professor>\n"
        f"<grounding>\n{json.dumps(grounding, indent=1)}\n</grounding>\n\n"
        "Write the email."
    )

    out = complete_json(SYSTEM_PROMPT, user, _SCHEMA)
    return out["subject"], out["body"]
