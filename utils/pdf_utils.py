import fitz  

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF and returns a single string 
    """
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text
