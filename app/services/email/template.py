"""Deterministic template generator: every sentence is filled from EmailContext. The baseline."""
from app.schemas import EmailContext, StudentProfile
from app.services.email.context import ProfessorInfo, get_salutation, oxford_join

GENERIC_SKILLS = {"C", "Java", "Linux", "Git"}


def _intro(student: StudentProfile) -> str:
    descriptor = " ".join(x for x in [student.year, student.major] if x)
    who = f"a {descriptor} student" if descriptor else "a student"
    if student.university:
        who += f" at {student.university}"
    return f"I am {student.name}, {who}. I am writing to express my interest in joining your research group."


def generate(prof: ProfessorInfo, student: StudentProfile, ctx: EmailContext) -> tuple[str, str]:
    """Returns (subject, body)."""
    # Interest line
    if ctx.best_interest and ctx.relevant_interests:
        interest_line = (f"I am particularly drawn to your work in {ctx.best_interest}, which aligns "
                         f"closely with my background in {oxford_join(ctx.relevant_interests[:2])}.")
    elif ctx.best_interest:
        interest_line = f"I am particularly drawn to your research in {ctx.best_interest}."
    else:
        interest_line = "I have been following your research and find it closely aligned with my interests."

    # Paper line
    paper_line = ""
    if ctx.paper_title:
        year = f"{ctx.paper_year} " if ctx.paper_year else ""
        paper_line = (f"I was particularly interested in your {year}work on \"{ctx.paper_title}\", "
                      f"which connects closely to my own projects in this area.")

    # Project line
    if ctx.best_project:
        project_line = f"Most recently, I worked on {ctx.best_project}, which strengthened my interest in this area."
    elif student.interests:
        project_line = (f"Through my coursework and projects in {oxford_join(student.interests[:2])}, "
                        f"I have developed a strong foundation relevant to your work.")
    else:
        project_line = "Through my coursework and projects, I have developed a strong foundation relevant to your work."

    # Skill line — drawn from the student's own skills, never hardcoded
    specific = [s for s in ctx.skill_overlap if s not in GENERIC_SKILLS]
    if len(specific) >= 2:
        skill_line = f"I have hands-on experience with {oxford_join(specific[:3])}, which I have applied across several projects."
    elif len(specific) == 1:
        skill_line = f"I have hands-on experience with {specific[0]}, among other tools relevant to your research."
    elif student.skills:
        skill_line = f"My technical background includes {oxford_join(student.skills[:4])}."
    else:
        skill_line = ""

    middle = [interest_line]
    if paper_line:
        middle.append(paper_line)
    middle.append(f"{project_line} {skill_line}".strip())

    subject = f"Prospective Research Student – {ctx.best_interest or 'Research Opportunities'}"
    signature = "\n".join(x for x in [student.name, student.email] if x)
    body = "\n\n".join([
        f"Dear Professor {get_salutation(prof.name)},",
        _intro(student),
        *middle,
        "I would welcome the opportunity to discuss how my background might contribute to your lab's work. "
        "I have attached my resume for your reference and am happy to meet at your convenience.",
        "Thank you for your time.",
        f"Best regards,\n{signature}",
    ])
    return subject, body
