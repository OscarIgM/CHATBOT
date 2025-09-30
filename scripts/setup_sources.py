import os
import csv
from pathlib import Path

# Rutas
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SOURCES_FILE = DATA_DIR / "sources.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

files = list(RAW_DIR.glob("*.pdf")) + list(RAW_DIR.glob("*.html")) + list(RAW_DIR.glob("*.csv"))

if not files:
    print("No se encontraron archivos PDF o HTML en data/raw/")
    exit(1)

# Generar sources.csv
with open(SOURCES_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["doc_id", "title", "source_type", "url", "vigencia", "filename"]
    )
    writer.writeheader()

    for i, file_path in enumerate(files, start=1):
        doc_id = f"doc_{i:03d}"
        title = file_path.stem.replace("_", " ")
        source_type = file_path.suffix.lower().replace(".", "")
        url = "https://www.ufro.cl"  
        vigencia = "" 
        filename = str(file_path).replace("\\", "/")  

        writer.writerow({
            "doc_id": doc_id,
            "title": title,
            "source_type": source_type,
            "url": url,
            "vigencia": vigencia,
            "filename": filename
        })

print(f"Archivo sources.csv generado en {SOURCES_FILE}")
