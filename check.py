import sqlite3

conn = sqlite3.connect("agentops.db")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in cur.fetchall():
    print(row[0])
conn.close()