import psycopg2
import sys
import re
sys.path.append(".")


def oxford_join(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def clean_interests(professor_interests):
    cleaned = []
    for interest in professor_interests:
        parts = [p.strip() for p in interest.split(",")]
        cleaned.extend(parts)
    return [i for i in cleaned if i]


def get_salutation(full_name):
    parts = full_name.strip().split()
    return parts[-1]


def get_skill_overlap(student_skills, professor_summary):
    summary_lower = professor_summary.lower()
    matches = []
    for skill in student_skills:
        if len(skill) <= 1:
            continue
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, summary_lower):
            matches.append(skill)
    return matches


def get_project_alignment(student_projects, professor_summary):
    summary_lower = professor_summary.lower()
    keywords = {
        "FPGA": ["fpga", "reconfigurable", "hardware acceleration", "hls"],
        "computer vision": ["vision", "image", "detection", "recognition", "visual"],
        "machine learning": ["machine learning", "deep learning", "neural", "ml", "ai"],
        "embedded systems": ["embedded", "firmware", "microcontroller", "hardware"],
        "signal processing": ["signal", "processing", "filtering", "acoustic"],
        "ASL/gestures": ["gesture", "hand", "interface", "human computer"],
    }
    aligned = []
    for project in student_projects:
        project_lower = project.lower()
        for topic, terms in keywords.items():
            if any(t in project_lower for t in terms):
                if any(t in summary_lower for t in terms):
                    aligned.append((project, topic))
                    break
    return aligned


def get_best_project(aligned_projects, professor_summary):
    if not aligned_projects:
        return None
    summary_lower = professor_summary.lower()
    best = None
    best_score = -1
    for project, topic in aligned_projects:
        words = topic.lower().split()
        score = sum(1 for w in words if w in summary_lower)
        if score > best_score:
            best_score = score
            best = project
    return best


def get_relevant_interests(student_interests, professor_summary):
    summary_lower = professor_summary.lower()
    return [interest for interest in student_interests
            if any(word in summary_lower for word in interest.lower().split())]


def get_best_interest(cleaned_prof_interests, student_interests):
    """Pick the single best professor interest that matches student background."""
    if not cleaned_prof_interests:
        return None
    student_text = " ".join(student_interests).lower()
    for interest in cleaned_prof_interests:
        words = [w for w in interest.lower().split() if len(w) > 3]
        if any(word in student_text for word in words):
            return interest
    return cleaned_prof_interests[0]


def get_relevant_paper(professor_email, student_embedding):
    conn = psycopg2.connect(dbname="research_outreach")
    cur = conn.cursor()
    cur.execute("""
        SELECT title, abstract, year,
               1 - (embedding <=> %s::vector) AS similarity
        FROM papers
        WHERE professor_email = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 1
    """, (student_embedding, professor_email, student_embedding))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return None
    title, abstract, year, similarity = row
    if similarity < 0.15:
        return None
    snippet = abstract[:150].rsplit(" ", 1)[0] + "..." if abstract else ""
    return {"title": title, "year": year, "similarity": similarity, "snippet": snippet}


def generate_email(professor_name, professor_email, professor_summary,
                   professor_interests, student_profile, student_embedding=None):

    skill_overlap = get_skill_overlap(student_profile["skills"], professor_summary)
    aligned_projects = get_project_alignment(student_profile["projects"], professor_summary)
    relevant_interests = get_relevant_interests(student_profile["interests"], professor_summary)
    cleaned_prof_interests = clean_interests(professor_interests)

    # Pick ONE best interest — used for both subject and body
    best_interest = get_best_interest(cleaned_prof_interests, student_profile["interests"])

    # --- Paper reference ---
    paper = None
    paper_line = ""
    if student_embedding:
        paper = get_relevant_paper(professor_email, student_embedding)
        if paper:
            paper_line = (f"I was particularly interested in your {paper['year']} work on "
                         f"\"{paper['title']}\", which connects closely to my own projects in this area.")

    # --- Project line ---
    best_project = get_best_project(aligned_projects, professor_summary)
    if best_project:
        project_line = f"Most recently, I worked on {best_project}, which strengthened my interest in this area."
    else:
        # Fallback — find any project with keyword overlap against professor summary
        summary_lower = professor_summary.lower()
        fallback = None
        for proj in student_profile["projects"]:
            if any(w in summary_lower for w in proj.lower().split() if len(w) > 4):
                fallback = proj
                break
        if fallback:
            project_line = f"Most recently, I worked on {fallback}, which strengthened my interest in this area."
        else:
            project_line = (f"Through my coursework and projects in "
                           f"{oxford_join(student_profile['interests'][:2])}, "
                           f"I have developed a strong foundation relevant to your work.")

    # --- Skill line ---
    if len(skill_overlap) >= 2:
        skill_line = f"I have hands-on experience with {oxford_join(skill_overlap[:3])}, which I have applied across several projects."
    elif len(skill_overlap) == 1 and skill_overlap[0] not in ["C", "Java"]:
        skill_line = f"I have hands-on experience with {skill_overlap[0]}, among other tools relevant to your research."
    else:
        skill_line = "My background spans Python, PyTorch, C++, and SystemVerilog, with a focus on applied ML and hardware systems."

    # --- Interest line — subject and body use same best_interest ---
    if best_interest:
        if relevant_interests:
            interest_line = (f"I am particularly drawn to your work in {best_interest}, "
                           f"which aligns closely with my background in {oxford_join(relevant_interests[:2])}.")
        else:
            interest_line = f"I am particularly drawn to your research in {best_interest}."
    else:
        interest_line = "I have been following your research and find it closely aligned with my interests."

    # Subject uses same best_interest
    subject_topic = best_interest if best_interest else "Research Collaboration"
    last_name = get_salutation(professor_name)

    # Assemble body
    if paper_line:
        body_middle = f"{interest_line}\n\n{paper_line}\n\n{project_line} {skill_line}"
    else:
        body_middle = f"{interest_line}\n\n{project_line} {skill_line}"

    email = f"""Subject: Prospective Research Student – {subject_topic}

Dear Professor {last_name},

I am {student_profile['name']}, a junior studying {student_profile['major']} at {student_profile['university']}. I am writing to express my interest in joining your research group.

{body_middle}

I would welcome the opportunity to discuss how my background might contribute to your lab's work. I have attached my resume for your reference and am happy to meet at your convenience.

Thank you for your time.

Best regards,
{student_profile['name']}
{student_profile['email']}"""

    return email.strip(), {
        "skill_overlap": skill_overlap,
        "aligned_projects": [p[0] for p in aligned_projects],
        "relevant_interests": relevant_interests,
        "paper_referenced": paper["title"] if paper else None
    }


def run_email_pipeline(pdf_path, top_k=3):
    from ranking.match import get_top_matches
    from ranking.reranker import rerank
    from sentence_transformers import SentenceTransformer
    from resume_parser.parse_resume import parse_resume, build_student_summary

    model = SentenceTransformer("all-MiniLM-L6-v2")
    profile = parse_resume(pdf_path)
    summary = build_student_summary(profile)
    student_embedding = model.encode(summary).tolist()

    _, matches = get_top_matches(pdf_path, top_k=116)
    reranked = rerank(matches, profile)
    top_matches = reranked[:top_k]

    conn = psycopg2.connect(dbname="research_outreach")
    cur = conn.cursor()

    for r in top_matches:
        name = r["name"]
        email = r["email"]
        summary = r["summary"]
        prof_interests = r["prof_interests"]

        generated_email, context = generate_email(
            name, email, summary, prof_interests,
            profile, student_embedding=student_embedding
        )

        print(f"{'='*60}")
        print(f"Professor: {name} | Score: {r['scores']['final']:.4f}")
        if context.get("paper_referenced"):
            print(f"Paper referenced: {context['paper_referenced']}")
        print(f"\n{generated_email}\n")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run_email_pipeline("ML_Resume.pdf", top_k=3)