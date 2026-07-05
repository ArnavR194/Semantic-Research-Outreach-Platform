import pdfplumber
import re
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# SECTION 1: RAW TEXT EXTRACTION
# ─────────────────────────────────────────

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)
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

# ─────────────────────────────────────────
# SECTION 5: LLM FALLBACK
# ─────────────────────────────────────────

def llm_parse_fallback(raw_text):
    """
    Called when DIY parser confidence is too low.
    Uses Anthropic API to extract structured profile.
    """
    import anthropic, json

    client = anthropic.Anthropic()
    prompt = f"""Extract structured information from this resume. 
Return ONLY valid JSON with no explanation, no markdown, no backticks.

Resume:
{raw_text}

Return exactly this format:
{{
  "name": "...",
  "email": "...",
  "skills": ["...", "..."],
  "coursework": ["...", "..."],
  "projects": ["...", "..."],
  "interests": ["...", "..."]
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

# ─────────────────────────────────────────
# SECTION 6: MAIN PIPELINE
# ─────────────────────────────────────────

def parse_resume(pdf_path, confidence_threshold=0.6):
    raw_text = extract_text(pdf_path)
    
    # Clean cid artifacts globally
    raw_text = re.sub(r"\(cid:\d+\)", "", raw_text)

    sections = detect_sections(raw_text)
    skills = parse_skills(sections)
    coursework = parse_coursework(sections)
    projects = parse_projects(sections)
    bullets = parse_experience_bullets(sections)
    interests = infer_interests(skills, projects, bullets)

    # Extract name and email from header
    header = sections.get("header", "")
    name = header.split("\n")[0].strip() if header else "Unknown"
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-z]+", header)
    email = email_match.group(0) if email_match else ""

    profile = {
        "name": name,
        "email": email,
        "university": "UIUC",
        "major": "Computer Engineering",
        "skills": skills,
        "coursework": coursework,
        "projects": projects,
        "interests": interests,
    }

    score = confidence_score(profile)
    print(f"Parser confidence: {score}")

    if score < confidence_threshold:
        print("Low confidence — falling back to LLM parser...")
        llm_profile = llm_parse_fallback(raw_text)
        # Merge — LLM fills gaps, DIY keeps what it got right
        for key in ["skills", "coursework", "projects", "interests"]:
            if not profile[key] and llm_profile.get(key):
                profile[key] = llm_profile[key]
        profile["name"] = llm_profile.get("name", profile["name"])
        profile["email"] = llm_profile.get("email", profile["email"])

    return profile

def build_student_summary(profile):
    parts = []
    if profile.get("skills"):
        parts.append("Skills: " + ", ".join(profile["skills"]))
    if profile.get("coursework"):
        parts.append("Coursework: " + ", ".join(profile["coursework"]))
    if profile.get("projects"):
        parts.append("Projects: " + " | ".join(profile["projects"]))
    if profile.get("interests"):
        parts.append("Interests: " + ", ".join(profile["interests"]))
    return " | ".join(parts)

def get_student_embedding(profile):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    summary = build_student_summary(profile)
    embedding = model.encode(summary).tolist()
    return embedding, summary

if __name__ == "__main__":
    import json
    profile = parse_resume("ML_Resume.pdf")
    print(json.dumps(profile, indent=2))
    print(f"\nConfidence: {confidence_score(profile)}")