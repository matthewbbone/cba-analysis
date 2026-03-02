import os
from pypdf import PdfReader
from dotenv import load_dotenv
from pathlib import Path
load_dotenv() # Load environment variables from .env file

def count_total_pdf_pages(folder_path):
    total = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                # Read PDF and get page count
                total += len(PdfReader(os.path.join(folder_path, filename)).pages)
            except Exception as e:
                print(f"Error reading {filename}: {e}") # Handles errors
    return total

# Usage: replace with your folder path
print(f"Total pages: {count_total_pdf_pages(Path(os.getenv('CACHE_DIR')) / "dol_archive")}")