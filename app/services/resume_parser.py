"""Hybrid resume parser: heuristic section parsing, with an LLM fallback when confidence is low.

Pure with respect to storage: takes PDF bytes (or a path), returns a StudentProfile + metadata.
"""
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from app.config import get_settings
from app.schemas import StudentProfile

log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# SECTION 1: RAW TEXT EXTRACTION
# ─────────────────────────────────────────

def extract_text(pdf: bytes | str | Path) -> str:
    source = io.BytesIO(pdf) if isinstance(pdf, (bytes, bytearray)) else pdf
    with pdfplumber.open(source) as doc:
        # extract_text() returns None for image-only pages
        text = "\n".join(page.extract_text() or "" for page in doc.pages)
    return text

# ─────────────────────────────────────────
# SECTION 2: SECTION DETECTION
# ─────────────────────────────────────────

SECTION_HEADERS = [
    "Education", "Relevant Coursework", "Experience",
    "Leadership", "Extracurricular", "Projects",
    "Technical Skills", "Skills", "Research", "Publications"
]

def detect_sections(text):
    """Split resume text into named sections."""
    lines = text.split("\n")
    sections = {}
    current_section = "header"
    buffer = []

    for line in lines:
        stripped = line.strip()
        # Check if this line is a section header
        matched = next(
            (h for h in SECTION_HEADERS if stripped.lower().startswith(h.lower())),
            None
        )
        if matched:
            sections[current_section] = "\n".join(buffer).strip()
            current_section = matched
            buffer = []
        else:
            buffer.append(line)

    sections[current_section] = "\n".join(buffer).strip()
    return sections

# ─────────────────────────────────────────
# SECTION 3: FIELD PARSERS
# ─────────────────────────────────────────

SKILLS_BLACKLIST = {
    "certified", "microsoft office", "fundamentals", "and", "the",
    "office", "suite", "advanced", "basic", "proficient", "familiar",
    "experience", "knowledge", "azure fundamentals"
}

def parse_skills(sections):
    raw = sections.get("Technical Skills", "") or sections.get("Skills", "")
    skills = []
    for line in raw.split("\n"):
        if ":" in line:
            items = line.split(":", 1)[1]
        else:
            items = line
        parsed = [s.strip() for s in items.split(",") if s.strip()]
        parsed = [s for s in parsed
                  if len(s) > 1
                  and s.lower() not in SKILLS_BLACKLIST
                  and not s.lower().startswith("microsoft")
                  and not s.lower().startswith("azure")]
        skills.extend(parsed)
    return list(dict.fromkeys(skills))


def parse_coursework(sections):
    """Extract courses from Relevant Coursework section."""
    raw = sections.get("Relevant Coursework", "")
    courses = []
    for line in raw.split("\n"):
        # Bullet points separated by • 
        parts = re.split(r"•", line)
        for part in parts:
            # Clean cid artifacts from PDF
            clean = re.sub(r"\(cid:\d+\)", "", part).strip()
            if clean and len(clean) > 3:
                courses.append(clean)
    return courses

def parse_projects(sections):
    raw = sections.get("Projects", "")
    projects = []
    for line in raw.split("\n"):
        stripped = line.strip()
        # Skip empty, bullets, and description lines
        if not stripped:
            continue
        if stripped.startswith("–") or stripped.startswith("-") or stripped.startswith("•"):
            continue
        # Skip lines that are clearly mid-sentence (start lowercase or with numbers)
        if stripped[0].islower():
            continue
        # Skip lines that look like descriptions (contain common description words)
        description_signals = ["improving", "extraction", "confusion", "localization",
                               "performance", "enabling", "optimizing", "ensuring",
                               "collaborating", "applying", "integrating"]
        if any(word in stripped.lower() for word in description_signals):
            continue
        if len(stripped) > 10:
            projects.append(stripped)
    return projects

def parse_experience_bullets(sections):
    """Extract bullet points from Experience and Leadership sections."""
    bullets = []
    for section_name in ["Experience", "Leadership"]:
        raw = sections.get(section_name, "")
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped.startswith("–") or stripped.startswith("-"):
                clean = stripped.lstrip("–- ").strip()
                if len(clean) > 20:
                    bullets.append(clean)
    return bullets

def infer_interests(skills, projects, bullets):
    text = " ".join(skills + projects + bullets).lower()

    interest_map = {
        # CS / AI / ML
        "machine learning": ["pytorch", "torchvision", "neural", "ml", "deep learning",
                             "classification", "detection", "molo", "hyperparameter",
                             "transformer", "fine-tuning", "inference", "training"],
        "computer vision": ["mediapipe", "vision", "image", "detection", "tracking",
                            "gesture", "hand", "visual", "camera", "opencv", "yolo",
                            "segmentation", "object detection", "feature extraction"],
        "AI systems": ["ai", "agent", "agentic", "autonomous", "asl", "recognition",
                       "llm", "generative", "nlp", "language model", "prompt"],
        "data science": ["pandas", "numpy", "matplotlib", "data analysis", "statistics",
                         "regression", "clustering", "sklearn", "jupyter"],

        # Hardware / Digital Design
        "FPGA and reconfigurable computing": ["fpga", "systemverilog", "vivado", "reconfigurable",
                                              "rtl", "hls", "verilog", "vhdl", "xilinx",
                                              "intel quartus", "bitstream", "synthesis"],
        "computer architecture": ["cpu", "microarchitecture", "pipeline", "cache", "risc",
                                  "mips", "arm", "out-of-order", "branch prediction",
                                  "memory hierarchy", "instruction set"],
        "digital circuits and VLSI": ["vlsi", "cmos", "logic design", "asic", "timing",
                                      "place and route", "cadence", "synopsys", "layout",
                                      "standard cell", "digital design"],
        "embedded systems": ["uart", "firmware", "microcontroller", "attiny", "arduino",
                             "stm32", "rtos", "pcb", "kicad", "c++", "real-time",
                             "bare metal", "i2c", "spi", "gpio", "interrupt"],

        # Analog / RF / Photonics
        "analog and mixed-signal circuits": ["analog", "op-amp", "amplifier", "filter",
                                             "adc", "dac", "oscillator", "comparator",
                                             "feedback", "cadence virtuoso", "spice",
                                             "mixed-signal", "bias"],
        "RF and wireless communications": ["rf amplifier", "antenna design", "wireless protocol",
                                    "mimo", "ofdm", "radar", "microwave",
                                    "transmission line", "impedance matching"],
        "signal processing": ["signal", "dsp", "fft", "filter", "convolution",
                              "fourier", "sampling", "noise", "matlab", "spectrogram"],
        "photonics and optoelectronics": ["photonics", "laser", "optical", "waveguide",
                                          "led", "photodetector", "fiber", "silicon photonics",
                                          "lidar", "optoelectronics"],

        # Semiconductor / Materials
        "semiconductor devices": ["semiconductor", "mosfet", "transistor", "diode",
                                  "bandgap", "doping", "fabrication", "cleanroom",
                                  "oxide", "diffusion", "implantation", "cmos process"],
        "semiconductor manufacturing": ["lithography", "etching", "deposition", "cvd",
                                        "pvd", "wafer", "process integration", "yield",
                                        "metrology", "inspection"],
        "materials science": ["materials", "crystal", "thin film", "nanostructure",
                              "2d materials", "graphene", "ferroelectric", "piezoelectric",
                              "magnetic", "characterization"],

        # Power / Energy
        "power electronics": ["power electronics", "converter", "inverter", "dc-dc",
                              "pwm", "mosfet driver", "igbt", "buck", "boost",
                              "motor drive", "switching"],
        "energy systems": ["renewable energy", "solar cell", "wind turbine", "battery management",
                   "electric vehicle", "smart grid", "power grid", "energy storage"],
        "electromechanics": ["motor", "actuator", "generator", "torque", "magnetic field",
                             "induction", "permanent magnet", "servo"],

        # Systems / Networking
        "software engineering": ["rest api", "spring boot", "java", "mongodb",
                                 "postgresql", "backend", "microservices", "docker",
                                 "kubernetes", "git", "ci/cd"],
        "computer networks": ["networking", "tcp", "udp", "routing", "protocol",
                              "socket", "latency", "bandwidth", "distributed", "p2p"],
        "cybersecurity": ["security", "cryptography", "encryption", "vulnerability",
                          "penetration", "firewall", "authentication", "zero trust"],
        "robotics and control": ["robotics", "ros", "control", "pid", "kalman",
                                 "localization", "slam", "path planning", "actuator",
                                 "sensor fusion"],
    }

    inferred = []
    for interest, keywords in interest_map.items():
        if any(kw in text for kw in keywords):
            inferred.append(interest)
    return inferred

# ─────────────────────────────────────────
# SECTION 4: CONFIDENCE SCORING
# ─────────────────────────────────────────

def confidence_score(profile):
    """
    Score how complete the parsed profile is (0.0 - 1.0).
    If below threshold, trigger LLM fallback.
    """
    score = 0
    if len(profile["skills"]) >= 5:
        score += 0.35
    elif len(profile["skills"]) >= 2:
        score += 0.15

    if len(profile["coursework"]) >= 3:
        score += 0.25
    elif len(profile["coursework"]) >= 1:
        score += 0.10

    if len(profile["projects"]) >= 1:
        score += 0.25

    if len(profile["interests"]) >= 2:
        score += 0.15

    return round(score, 2)


def parse_header_details(raw_text: str) -> dict:
    """Best-effort university / major / class year from the education block."""
    out: dict = {"university": None, "major": None, "year": None}
    month = r"(?!(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b)"
    cap_word = month + r"[A-Z][\w\-]*"
    uni = re.search(
        rf"(University of(?: {cap_word})+(?: at(?: {cap_word})+)?"
        rf"|{cap_word}(?: {cap_word})* (?:University|Institute of Technology|College))",
        raw_text,
    )
    if uni:
        out["university"] = uni.group(1).strip()
    major = re.search(
        r"(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|Ph\.?D\.?|Bachelor of \w+|Master of \w+)(?:\s+(?:in|of))?\s+([A-Z][A-Za-z&]+(?: (?:and |& )?[A-Z][A-Za-z&]+)*)",
        raw_text,
    )
    if major:
        out["major"] = major.group(1).strip()
    year = re.search(r"\b(freshman|sophomore|junior|senior)\b", raw_text, re.IGNORECASE)
    if year:
        out["year"] = year.group(1).lower()
    return out

# ─────────────────────────────────────────
# SECTION 5: LLM FALLBACK
# ─────────────────────────────────────────

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "university": {"type": "string"},
        "major": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "coursework": {"type": "array", "items": {"type": "string"}},
        "projects": {"type": "array", "items": {"type": "string"}},
        "interests": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "email", "university", "major", "skills", "coursework", "projects", "interests"],
    "additionalProperties": False,
}


def llm_parse_fallback(raw_text: str) -> dict:
    """Structured extraction with the configured LLM. Called when heuristic confidence is low."""
    from app.services.llm_client import complete_json

    return complete_json(
        system=(
            "You extract structured data from student resumes. Use an empty string or empty list "
            "for anything the resume does not state. 'projects' are project titles, one per item. "
            "'interests' are research areas the resume shows evidence of, e.g. 'computer vision'."
        ),
        user=f"<resume>\n{raw_text}\n</resume>",
        schema=_LLM_SCHEMA,
    )

# ─────────────────────────────────────────
# SECTION 6: MAIN PIPELINE
# ─────────────────────────────────────────

@dataclass
class ParseResult:
    profile: StudentProfile
    confidence: float
    used_llm: bool


def parse_resume(pdf: bytes | str | Path, confidence_threshold: float | None = None) -> ParseResult:
    s = get_settings()
    threshold = s.parser_confidence_threshold if confidence_threshold is None else confidence_threshold

    raw_text = extract_text(pdf)
    raw_text = re.sub(r"\(cid:\d+\)", "", raw_text)  # PDF glyph artifacts

    sections = detect_sections(raw_text)
    skills = parse_skills(sections)
    coursework = parse_coursework(sections)
    projects = parse_projects(sections)
    bullets = parse_experience_bullets(sections)
    interests = infer_interests(skills, projects, bullets)

    header = sections.get("header", "")
    name = header.split("\n")[0].strip() if header else "Unknown"
    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", raw_text)

    profile = {
        "name": name or "Unknown",
        "email": email_match.group(0) if email_match else "",
        **parse_header_details(raw_text),
        "skills": skills,
        "coursework": coursework,
        "projects": projects,
        "interests": interests,
    }

    score = confidence_score(profile)
    used_llm = False
    log.info("Parser confidence: %s", score)

    if score < threshold and s.llm_enabled:
        log.info("Low confidence; falling back to LLM parser")
        try:
            llm_profile = llm_parse_fallback(raw_text)
            used_llm = True
            # LLM fills gaps; heuristic keeps what it already got
            for key in ["skills", "coursework", "projects", "interests"]:
                if not profile[key] and llm_profile.get(key):
                    profile[key] = llm_profile[key]
            for key in ["name", "email", "university", "major"]:
                if llm_profile.get(key) and (not profile.get(key) or profile[key] == "Unknown"):
                    profile[key] = llm_profile[key]
        except Exception:
            log.exception("LLM fallback failed; keeping heuristic profile")

    return ParseResult(StudentProfile(**profile), score, used_llm)


def build_student_summary(profile: StudentProfile) -> str:
    parts = []
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.coursework:
        parts.append("Coursework: " + ", ".join(profile.coursework))
    if profile.projects:
        parts.append("Projects: " + " | ".join(profile.projects))
    if profile.interests:
        parts.append("Interests: " + ", ".join(profile.interests))
    return " | ".join(parts)
