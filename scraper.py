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


def scrape_job_postings(url="https://realpython.github.io/fake-jobs/", limit=5):
    """
    Scrape job postings from a static, scraping-permitted job board.
    Uses a practice site built for scraping (no ToS/anti-bot issues).
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
        jobs.append({
            "title": title.get_text(strip=True),
            "company": company.get_text(strip=True) if company else "",
            "location": location.get_text(strip=True) if location else "",
            "url": url,
            "description": f"{title.get_text(strip=True)} at {company.get_text(strip=True) if company else 'unknown'}. Scraped listing."
        })
    return jobs


def import_scraped_jobs(url="https://realpython.github.io/fake-jobs/", limit=5):
    jobs = scrape_job_postings(url, limit)
    if not jobs:
        print("No jobs scraped.")
        return

    conn = get_connection()
    cur = conn.cursor()
    # idempotent: clear previous scraped rows
    cur.execute("DELETE FROM job_postings WHERE source = 'scraped'")

    for job in jobs:
        cur.execute("""
            INSERT INTO job_postings (title, company, description, location, work_mode, source)
            VALUES (%s, %s, %s, %s, %s, 'scraped')
        """, (job["title"], job["company"], job["description"], job["location"], ""))

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM job_postings")
    total = cur.fetchone()[0]
    conn.close()
    print(f"Scraped and imported {len(jobs)} jobs. Total in table: {total}")


if __name__ == "__main__":
    import_scraped_jobs()