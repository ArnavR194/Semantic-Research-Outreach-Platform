"""Scrape the UIUC ECE faculty directory into data/professors.json.

    python -m pipelines.crawl
"""
import requests
from bs4 import BeautifulSoup
import json
import time
import os

from app.config import DATA_DIR

RAW_HTML_DIR = DATA_DIR / "raw_html"

BASE_URL = "https://ece.illinois.edu"
DIRECTORY_URL = f"{BASE_URL}/about/directory/faculty-dept"

def get_faculty_links(directory_url):
    resp = requests.get(directory_url, timeout=15)
    soup = BeautifulSoup(resp.text, "lxml")

    faculty = []
    for a in soup.select("a[href*='/about/directory/faculty/']"):  # note: *=  not ^=
        href = a["href"]

        # Skip the index page link itself
        if href == "/about/directory/faculty-dept":
            continue

        name = a.get_text(strip=True)
        if not name:
            continue

        url = BASE_URL + href

        # Grab title — it's in a <p> or text node right after the <a>
        parent = a.find_parent()
        siblings = list(parent.next_siblings)
        title = "Unknown"
        for sib in siblings:
            text = sib.get_text(strip=True) if hasattr(sib, "get_text") else str(sib).strip()
            if any(k in text for k in ["Professor", "Lecturer", "Instructor"]):
                title = text
                break

        faculty.append({"name": name, "title": title, "url": url})

    return faculty

def parse_professor_page(prof):
    url = prof["url"]
    try:
        resp = requests.get(url, timeout=10)
    except Exception as e:
        print(f"  Failed: {url}: {e}")
        return None

    # Save raw HTML
    slug = url.split("/")[-1]
    os.makedirs(RAW_HTML_DIR, exist_ok=True)
    with open(RAW_HTML_DIR / f"{slug}.html", "w", encoding="utf-8") as f:
        f.write(resp.text)

    soup = BeautifulSoup(resp.text, "lxml")
    main = soup.select_one("main")
    if not main:
        return None

    def get_section_items(heading_text):
        """Find an h2 by name, return all <li> text under it until next h2."""
        for h2 in main.find_all("h2"):
            if heading_text.lower() in h2.get_text(strip=True).lower():
                items = []
                for sib in h2.next_siblings:
                    if sib.name == "h2":
                        break
                    if sib.name == "ul":
                        items += [li.get_text(strip=True) for li in sib.find_all("li")]
                    elif sib.name == "li":
                        items.append(sib.get_text(strip=True))
                return items
        return []

    def get_section_text(heading_text):
        """Find an h2 by name, return ALL <p> text under it until next h2."""
        for h2 in main.find_all("h2"):
            if heading_text.lower() in h2.get_text(strip=True).lower():
                paragraphs = []
                for sib in h2.next_siblings:
                    if sib.name == "h2":
                        break
                    if sib.name == "p":
                        text = sib.get_text(separator=" ", strip=True)
                        if text:
                            paragraphs.append(text)
                return " ".join(paragraphs) if paragraphs else None
        return None

    # Email — look only inside <main>, not the whole page
    email = None
    email_tag = main.find("a", href=lambda h: h and "mailto:" in h)
    if email_tag:
        email = email_tag["href"].replace("mailto:", "").strip()

    return {
        "name": prof["name"],
        "title": prof["title"],
        "url": url,
        "university": "UIUC",
        "department": "ECE",
        "email": email,
        "research_interests": get_section_items("Research Interests"),
        "research_areas": get_section_items("Research Areas"),
        "research_topics": get_section_items("Research Topics"),
        "bio": get_section_text("Biography"),
    }

def crawl(limit=116):
    print("Fetching faculty directory...")
    faculty_list = get_faculty_links(DIRECTORY_URL)
    print(f"Found {len(faculty_list)} faculty. Crawling first {limit}...")

    results = []
    for prof in faculty_list[:limit]:
        print(f"  -> {prof['name']}")
        data = parse_professor_page(prof)
        if data:
            results.append(data)
        time.sleep(1.5)  # Be respectful — don't hammer their server

    return results

if __name__ == "__main__":
    professors = crawl(limit=116)

    with open(DATA_DIR / "professors.json", "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(professors)} professors to professors.json")