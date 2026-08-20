import requests
from bs4 import BeautifulSoup
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def infer_employment_type(title, description):
    """
    Best-effort inference of employment type from text, since the scraped site
    doesn't provide it structurally. Defaults to full-time when nothing matches.
    """
    text = f"{title} {description}".lower()
    if "intern" in text:
        return "internship"
    if "part-time" in text or "part time" in text:
        return "part-time"
    if "contract" in text or "contractor" in text or "freelance" in text:
        return "contract"
    return "full-time"


def scrape_job_postings(url="https://realpython.github.io/fake-jobs/", limit=5):
    """
    Scrape job postings from a static, scraping-permitted practice site.
    """
    headers = {"User-Agent": "AgentOpsMonitor/1.0 (educational project)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    cards = soup.select("div.card-content")[:limit]

    jobs = []
    for card in cards:
        title = card.select_one("h2.title")
        company = card.select_one("h3.subtitle")
        location = card.select_one("p.location")
        if not title:
            continue
        t = title.get_text(strip=True)
        c = company.get_text(strip=True) if company else ""
        desc = f"{t} at {c if c else 'unknown'}. Scraped listing."
        jobs.append({
            "title": t,
            "company": c,
            "location": location.get_text(strip=True) if location else "",
            "url": url,
            "description": desc,
            "employment_type": infer_employment_type(t, desc)
        })
    return jobs


def import_scraped_jobs(url="https://realpython.github.io/fake-jobs/", limit=5):
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
    print(f"Scraped and imported {len(jobs)} jobs. Total in table: {total}")


if __name__ == "__main__":
    import_scraped_jobs()