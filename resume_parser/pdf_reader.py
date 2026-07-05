import pdfplumber

with pdfplumber.open("ML_Resume.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)

for i, line in enumerate(text.split("\n")):
    print(f"{i:3}: {line}")