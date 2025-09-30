import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
from pypdf import PdfReader
import csv

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOURCES_FILE = DATA_DIR / "sources.csv"

def clean_text(text: str) -> str:
    """Normaliza espacios y elimina saltos de línea excesivos."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size=900, overlap=120):
    """Divide el texto en chunks de aproximadamente chunk_size palabras con overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

sources_df = pd.read_csv(SOURCES_FILE)

records = []

files = list(RAW_DIR.glob("*.*"))
for file_path in tqdm(files, desc="Procesando archivos"):
    source_row = sources_df[sources_df['filename'].str.endswith(file_path.name)]
    if not source_row.empty:
        metadata_base = {
            "doc_id": source_row.iloc[0]["doc_id"],
            "title": source_row.iloc[0]["title"],
            "source_type": source_row.iloc[0]["source_type"],
            "url": source_row.iloc[0]["url"],
            "vigencia": source_row.iloc[0]["vigencia"],
            "filename": source_row.iloc[0]["filename"]
        }
    else:
        metadata_base = {"doc_id": file_path.stem, "title": file_path.stem}

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            for i, chunk in enumerate(chunk_text(text)):
                metadata = metadata_base.copy()
                metadata.update({"page": page_num, "chunk_id": f"{file_path.stem}_{page_num}_{i}"})
                records.append({
                    "doc_id": metadata["doc_id"],
                    "chunk_id": metadata["chunk_id"],
                    "text": chunk,
                    "metadata": metadata
                })

    elif file_path.suffix.lower() in [".csv", ".md", ".txt"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = clean_text(f.read())
            for i, chunk in enumerate(chunk_text(content)):
                metadata = metadata_base.copy()
                metadata.update({"chunk_id": f"{file_path.stem}_0_{i}"})
                records.append({
                    "doc_id": metadata["doc_id"],
                    "chunk_id": metadata["chunk_id"],
                    "text": chunk,
                    "metadata": metadata
                })
        except Exception as e:
            print(f"⚠️ Error leyendo {file_path}: {e}")

    else:
        print(f"⚠️ Tipo de archivo no soportado: {file_path.suffix}")

chunks_df = pd.DataFrame(records)
chunks_file = PROCESSED_DIR / "chunks.parquet"
chunks_df.to_parquet(chunks_file, index=False)

print(f"✅ Ingesta completada: {len(chunks_df)} chunks generados en {chunks_file}")

