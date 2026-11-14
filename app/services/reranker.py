"""Multi-signal reranker. Pure functions: no DB access, so the API and offline eval share it."""
from dataclasses import dataclass, field

from app.config import RerankWeights

# Tool → research domains it's evidence for. Generic tools map to [] and never count.
SKILL_DOMAIN_MAP: dict[str, list[str]] = {
    "PyTorch":       ["deep learning", "neural network", "machine learning framework"],
    "TensorFlow":    ["deep learning", "neural network", "machine learning framework"],
    "SystemVerilog": ["fpga", "vlsi", "rtl", "verilog", "hardware synthesis"],
    "Verilog":       ["fpga", "vlsi", "rtl", "verilog", "hardware synthesis"],
    "VHDL":          ["fpga", "vlsi", "rtl", "hardware synthesis"],
    "Vivado":        ["fpga", "xilinx", "reconfigurable computing"],
    "NumPy":         ["numerical computing", "signal processing", "scientific computing"],
    "MATLAB":        ["signal processing", "control system", "scientific computing"],
    "KiCad":         ["pcb design", "circuit board", "hardware design"],
    "Cadence":       ["vlsi", "analog circuit", "mixed-signal", "integrated circuit"],
    "SPICE":         ["analog circuit", "circuit simulation", "integrated circuit"],
    "ROS":           ["robotics", "robot", "autonomous vehicle"],
    "OpenCV":        ["computer vision", "image processing"],
    "CUDA":          ["gpu", "parallel computing", "high performance computing"],
    "C++":           ["real-time system", "embedded software", "robotics", "systems programming"],
    "AutoCAD":       ["cad", "mechanical design", "manufacturing"],
    "SolidWorks":    ["mechanical design", "cad", "manufacturing"],
    "Python": [], "Java": [], "JavaScript": [], "Spring Boot": [], "MongoDB": [],
    "PostgreSQL": [], "Linux": [], "Git": [], "C": [],
}

# Student interest (from the parser's interest map) → phrases a professor must explicitly state.
INTEREST_KEYWORD_MAP: dict[str, list[str]] = {
    "machine learning":                  ["machine learning", "deep learning", "neural network", "reinforcement learning"],
    "computer vision":                   ["computer vision", "object detection", "image segmentation", "visual recognition"],
    "AI systems":                        ["ai system", "artificial intelligence system", "autonomous system"],
    "data science":                      ["data science", "data analytics"],
    "FPGA and reconfigurable computing": ["fpga", "reconfigurable computing", "high-level synthesis", "hardware synthesis"],
    "computer architecture":             ["computer architecture", "microarchitecture", "memory system"],
    "digital circuits and VLSI":         ["vlsi", "digital circuit", "integrated circuit"],
    "embedded systems":                  ["embedded system", "firmware", "bare metal", "microcontroller"],
    "analog and mixed-signal circuits":  ["analog", "mixed-signal", "integrated circuit"],
    "RF and wireless communications":    ["wireless", "radio frequency", "antenna", "communication system"],
    "signal processing":                 ["signal processing", "image processing", "speech"],
    "photonics and optoelectronics":     ["photonic", "optic", "laser", "optoelectronic"],
    "semiconductor devices":             ["semiconductor device", "semiconductor", "nanoelectronic"],
    "semiconductor manufacturing":       ["fabrication", "nanofabrication", "manufacturing"],
    "materials science":                 ["material", "thin film", "nanostructure"],
    "power electronics":                 ["power electronic", "converter", "motor drive"],
    "energy systems":                    ["power system", "energy", "smart grid", "renewable"],
    "electromechanics":                  ["electromechanic", "electric machine", "motor"],
    "software engineering":              ["software system", "software engineering", "distributed system"],
    "computer networks":                 ["network", "internet", "distributed system"],
    "cybersecurity":                     ["security", "privacy", "cryptograph"],
    "robotics and control":              ["robotics", "robot", "autonomous vehicle", "control system", "control theory"],
}

# Course → phrases that indicate relevance. Unknown courses fall back to their full name.
COURSE_KEYWORD_MAP: dict[str, list[str]] = {
    "Analog Signal Processing":       ["analog", "signal processing", "filter"],
    "Data Structures and Algorithms": ["algorithm", "computational complexity"],
    "Computer Systems":               ["computer architecture", "operating system", "memory"],
    "Semiconductor Electronics":      ["semiconductor", "device", "mosfet", "cmos"],
    "Digital Systems Laboratory":     ["digital", "fpga", "logic", "verilog"],
    "Programming Methods for ML":     ["machine learning", "neural", "deep learning"],
}


@dataclass
class Candidate:
    """A professor returned by vector search, with everything the reranker needs."""
    professor_id: int
    name: str
    email: str | None
    title: str | None
    url: str | None
    research_summary: str
    research_terms: list[str]  # interests + areas + topics, flattened
    embedding_score: float


@dataclass
class Ranked:
    candidate: Candidate
    scores: dict[str, float]
    matches: dict[str, list[str]] = field(default_factory=dict)


def skill_overlap_score(student_skills: list[str], prof_terms: list[str]) -> tuple[float, list[str]]:
    """Count a skill only if it maps to a specific domain the professor explicitly works in."""
    prof_text = " ".join(prof_terms).lower()
    matches = [
        skill for skill in student_skills
        if (domains := SKILL_DOMAIN_MAP.get(skill)) and any(d in prof_text for d in domains)
    ]
    return min(len(matches) / 3.0, 1.0), matches


def interest_alignment_score(student_interests: list[str], prof_terms: list[str]) -> tuple[float, list[str]]:
    """Count an interest only on a full phrase match against professor-stated research."""
    prof_text = " ".join(prof_terms).lower()
    matches = [
        interest for interest in student_interests
        if (kws := INTEREST_KEYWORD_MAP.get(interest)) and any(kw in prof_text for kw in kws)
    ]
    return min(len(matches) / 4.0, 1.0), matches


def coursework_relevance_score(student_coursework: list[str], professor_summary: str) -> tuple[float, list[str]]:
    summary_lower = professor_summary.lower()
    matches = [
        course for course in student_coursework
        if any(kw in summary_lower for kw in COURSE_KEYWORD_MAP.get(course, [course.lower()]))
    ]
    return min(len(matches) / 3.0, 1.0), matches


def rerank(candidates: list[Candidate], profile, weights: RerankWeights | None = None) -> list[Ranked]:
    w = weights or RerankWeights()
    ranked = []
    for c in candidates:
        s_skill, skill_m = skill_overlap_score(profile.skills, c.research_terms)
        s_int, int_m = interest_alignment_score(profile.interests, c.research_terms)
        s_course, course_m = coursework_relevance_score(profile.coursework, c.research_summary)

        final = (w.embedding * c.embedding_score + w.skills * s_skill
                 + w.interests * s_int + w.coursework * s_course)
        ranked.append(Ranked(
            candidate=c,
            scores={
                "final": round(final, 4),
                "embedding": round(c.embedding_score, 4),
                "skills": round(s_skill, 4),
                "interests": round(s_int, 4),
                "coursework": round(s_course, 4),
            },
            matches={"skills": skill_m, "interests": int_m, "coursework": course_m},
        ))
    return sorted(ranked, key=lambda r: r.scores["final"], reverse=True)
