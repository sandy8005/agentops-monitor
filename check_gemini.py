import os
from dotenv import load_dotenv
load_dotenv()

print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
try:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-flash-latest",
        contents="Say the word OK."
    )
    print("SUCCESS:", resp.text)
except Exception as e:
    print("FAILED:", type(e).__name__, "-", e)