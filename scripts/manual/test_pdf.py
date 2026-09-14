# test_pdf.py
from pdf_reader import read_resume_file

text = read_resume_file("SANDEEP_BARIGE_Resume.pdf")
print(text[:1500])
print(f"\n--- total characters: {len(text)} ---")