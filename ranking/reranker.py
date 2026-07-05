import re
import psycopg2

def skill_overlap_score(student_skills, prof_interests):
    """
    Only count a skill match if it maps to a highly specific domain
    that the professor explicitly works in.
    """
    skill_domain_map = {
        # These should only match professors who explicitly work with these tools/domains
        "PyTorch":       ["deep learning", "neural network", "machine learning framework"],
        "SystemVerilog": ["fpga", "vlsi", "rtl", "verilog", "hardware synthesis"],
        "Vivado":        ["fpga", "xilinx", "reconfigurable computing"],
        "NumPy":         ["numerical computing", "signal processing", "scientific computing"],
        "KiCad":         ["pcb design", "circuit board", "hardware design"],
        "C++":           ["real-time system", "embedded software", "robotics", "systems programming"],
        "AutoCAD":       ["cad", "mechanical design", "manufacturing"],
        "SolidWorks":    ["mechanical design", "cad", "manufacturing"],
        # Generic skills — don't count these toward overlap at all
        "Python":        [],
        "Java":          [],
        "JavaScript":    [],
        "Spring Boot":   [],
        "MongoDB":       [],
        "PostgreSQL":    [],
        "Linux":         [],
    }

    prof_text = " ".join(prof_interests).lower()
    matches = []

    for skill in student_skills:
        domains = skill_domain_map.get(skill, [])
        if not domains:
            continue  # skip generic skills entirely
        if any(domain in prof_text for domain in domains):
            matches.append(skill)

    # Normalize over 3 instead of 5 — fewer specific skills possible
    return min(len(matches) / 3.0, 1.0), matches


def interest_alignment_score(student_interests, prof_interests):
    """
    Only count an interest match if the professor explicitly states
    that specific research area — not just tangentially related words.
    """
    interest_keyword_map = {
        "machine learning":                  ["machine learning", "deep learning", "neural network", "reinforcement learning"],
        "computer vision":                   ["computer vision", "object detection", "image segmentation", "visual recognition"],
        "AI systems":                        ["ai system", "artificial intelligence system", "autonomous system"],
        "data science":                      ["data science", "data analytics"],
        "FPGA and reconfigurable computing": ["fpga", "reconfigurable computing", "high-level synthesis", "hardware synthesis"],
        "embedded systems":                  ["embedded system", "firmware", "bare metal", "microcontroller"],
        "software engineering":              ["software system", "software engineering", "distributed system"],
        "robotics and control":              ["robotics", "robot", "autonomous vehicle", "control system"],
    }

    prof_text = " ".join(prof_interests).lower()
    matches = []

    for interest in student_interests:
        keywords = interest_keyword_map.get(interest, [])
        if not keywords:
            continue
        # Require full phrase match — not just a single word
        if any(kw in prof_text for kw in keywords):
            matches.append(interest)

    # Normalize over 4 — expect fewer exact matches
    return min(len(matches) / 4.0, 1.0), matches

def coursework_relevance_score(student_coursework, professor_summary):
    """Score based on coursework relevance to professor research."""
    summary_lower = professor_summary.lower()
    
    # Map courses to keywords that might appear in professor summaries
    course_keywords = {
        "Analog Signal Processing": ["analog", "signal processing", "filter"],
        "Data Structures and Algorithms": ["algorithm", "computational complexity"],
        "Computer Systems": ["computer architecture", "systems", "memory"],
        "Semiconductor Electronics": ["semiconductor", "device", "mosfet", "cmos"],
        "Digital Systems Laboratory": ["digital", "fpga", "logic", "verilog"],
        "Programming Methods for ML": ["machine learning", "neural", "deep learning"],
    }
    
    matches = []
    for course in student_coursework:
        keywords = course_keywords.get(course, [course.lower().split()])
        if any(kw in summary_lower for kw in keywords):
            matches.append(course)
    
    return min(len(matches) / 3.0, 1.0), matches

def rerank(matches, student_profile, weights=None):
    if weights is None:
        weights = {
            "embedding":  0.45,
            "skills":     0.25,
            "interests":  0.20,
            "coursework": 0.10,
        }

    conn = psycopg2.connect(dbname="research_outreach")
    cur = conn.cursor()

    reranked = []
    for name, email, summary, embedding_score in matches:
        cur.execute("""
            SELECT research_interests, research_areas, research_topics
            FROM professors WHERE email = %s
        """, (email,))
        row = cur.fetchone()
        prof_interests = []
        if row:
            for col in row:
                if col:
                    prof_interests.extend(col)

        # Updated function signatures
        s_skill, skill_matches = skill_overlap_score(student_profile["skills"], prof_interests)
        
        s_interest, interest_matches = interest_alignment_score(student_profile["interests"], prof_interests)

        
        s_course, course_matches = coursework_relevance_score(
            student_profile["coursework"], summary)

        final_score = (
            weights["embedding"]  * embedding_score +
            weights["skills"]     * s_skill +
            weights["interests"]  * s_interest +
            weights["coursework"] * s_course
        )

        reranked.append({
            "name": name,
            "email": email,
            "summary": summary,
            "prof_interests": prof_interests,
            "scores": {
                "final":      round(final_score, 4),
                "embedding":  round(embedding_score, 4),
                "skills":     round(s_skill, 4),
                "interests":  round(s_interest, 4),
                "coursework": round(s_course, 4),
            },
            "matches": {
                "skills":     skill_matches,
                "interests":  interest_matches,
                "coursework": course_matches,
            }
        })

    cur.close()
    conn.close()
    return sorted(reranked, key=lambda x: x["scores"]["final"], reverse=True)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from resume_parser.parse_resume import parse_resume
    from ranking.match import get_top_matches

    profile, matches = get_top_matches("ML_Resume.pdf", top_k=116)
    reranked = rerank(matches, profile)

    print(f"Reranked matches for {profile['name']}:\n")
    for i, r in enumerate(reranked[:10], 1):
        s = r["scores"]
        print(f"{i}. {r['name']}")
        print(f"   Final: {s['final']} | Embedding: {s['embedding']} | "
              f"Skills: {s['skills']} | Interests: {s['interests']} | "
              f"Coursework: {s['coursework']}")
        print(f"   Skill matches:     {r['matches']['skills']}")
        print(f"   Interest matches:  {r['matches']['interests']}")
        print(f"   Course matches:    {r['matches']['coursework']}")
        print()
    # Add this at the bottom of reranker.py temporarily