import csv
import sys
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def import_jobs_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM job_postings WHERE source = 'csv'")

    imported = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"title", "company", "description"}
        if not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames)
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            if not row["title"] or not row["description"]:
                print(f"  skipping row with empty title/description")
                continue
            # employment_type is optional in the CSV; default to full-time.
            emp_type = row.get("employment_type", "") or "full-time"
            cur.execute("""
                INSERT INTO job_postings (title, company, description, location, work_mode, employment_type, source)
                VALUES (%s, %s, %s, %s, %s, %s, 'csv')
            """, (
                row["title"], row.get("company", ""), row["description"],
                row.get("location", ""), row.get("work_mode", ""), emp_type
            ))
            imported += 1

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM job_postings")
    total = cur.fetchone()[0]
    conn.close()
    print(f"Imported {imported} jobs from CSV. Total in table: {total}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_jobs.csv"
    import_jobs_from_csv(path)