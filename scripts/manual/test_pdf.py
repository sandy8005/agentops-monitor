import sys
# test_pdf.py
from pdf_reader import read_resume_file

text = read_resume_file(sys.argv[1] if len(sys.argv) > 1 else "sample_resume.pdf")  # pass a SYNTHETIC resume path
print(text[:1500])
print(f"\n--- total characters: {len(text)} ---")