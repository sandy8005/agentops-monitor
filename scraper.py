import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = "https://realpython.github.io/fake-jobs/"


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def infer_employment_type(title, description):
    """Best-effort inference; the site doesn't structure this. Defaults to full-time."""
    text = f"{title} {description}".lower()
    if "intern" in text:
        return "internship"
    if "part-time" in text or "part time" in text:
        return "part-time"
    if "contract" in text or "contractor" in text or "freelance" in text:
        return "contract"
    return "full-time"


def fetch_detail_description(detail_url, headers, fallback):
    """
    Follow a listing's detail-page link and extract the real job description.
    Degrades gracefully: on ANY failure, return the fallback text so one bad
    detail page never breaks the whole scrape.
    """
    try:
        resp = requests.get(detail_url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # On the fake-jobs detail pages, the description sits in the content column.
        content = soup.select_one("div.content")
        if content:
            # Grab the paragraph text (the actual description body).
            paras = [p.get_text(strip=True) for p in content.select("p")]
            text = " ".join(t for t in paras if t)
            if text:
                return text
        # Fallback: whole-page text if the expected structure isn't found.
        body = soup.get_text(separator=" ", strip=True)
        return body[:2000] if body else fallback
    except Exception as e:
        print(f"    (detail fetch failed for {detail_url}: {e} — using listing text)")
        return fallback


def scrape_job_postings(url=BASE_URL, limit=5):
    """
    Scrape job postings from a static, scraping-permitted practice site.
    For each listing, follows the detail-page link to get the REAL description
    (not just the card title), so extracted requirements are meaningful.
    """
    headers = {"User-Agent": "AgentOpsMonitor/1.0 (educational project)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    cards = soup.select("div.card-content")[:limit]

    jobs = []
    for card in cards:
        title_el = card.select_one("h2.title")
        company_el = card.select_one("h3.subtitle")
        location_el = card.select_one("p.location")
        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        company = company_el.get_text(strip=True) if company_el else ""
        location = location_el.get_text(strip=True) if location_el else ""

        # Find the detail-page link ("Apply" / footer link on the card).
        detail_link = None
        for a in card.select("a"):
            href = a.get("href", "")
            if href and href.endswith(".html"):
                detail_link = urljoin(url, href)
                break

        fallback = f"{title} at {company if company else 'unknown'}."
        if detail_link:
            description = fetch_detail_description(detail_link, headers, fallback)
            time.sleep(0.5)   # polite delay between detail-page fetches
        else:
            description = fallback

        jobs.append({
            "title": title,
            "company": company,
            "location": location,
            "url": detail_link or url,
            "description": description,
            "employment_type": infer_employment_type(title, description)
        })
    return jobs


def import_scraped_jobs(url=BASE_URL, limit=5):
    jobs = scrape_job_postings(url, limit)
    if not jobs:
        print("No jobs scraped.")
        return

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM job_postings WHERE source = 'scraped'")

    for job in jobs:
        cur.execute("""
            INSERT INTO job_postings (title, company, description, location, work_mode, employment_type, source)
            VALUES (%s, %s, %s, %s, %s, %s, 'scraped')
        """, (job["title"], job["company"], job["description"], job["location"], "",
              job["employment_type"]))

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM job_postings")
    total = cur.fetchone()[0]
    conn.close()
    print(f"Scraped and imported {len(jobs)} jobs (with real descriptions). Total in table: {total}")


if __name__ == "__main__":
    import_scraped_jobs()