import fitz
doc = fitz.open("data/250883_english_01042024.pdf")
# This prints the first 1000 characters of the PDF as the computer sees them
print(doc[0].get_text("text")[:1000])