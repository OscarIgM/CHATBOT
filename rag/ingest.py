import pandas as pd
from pathlib import Path
from pypdf import PdfReader
import re
from tqdm import tqdm

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
    """Divide texto en fragmentos con solapamiento de palabras."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

sources = pd.read_csv(SOURCES_FILE)

records = []

for _, row in tqdm(sources.iterrows(), total=len(sources), desc="Procesando PDFs"):
    doc_id = row["doc_id"]
    title = row["title"]
    url = row.get("url", "")
    vigencia = row.get("vigencia", "")
    file_path = Path(row["filename"])

    if not file_path.exists():
        print(f"⚠️ Archivo no encontrado: {file_path}")
        continue

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            for i, chunk in enumerate(chunk_text(text)):
                records.append({
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "vigencia": vigencia,
                    "page": page_num,
                    "chunk_id": f"{doc_id}_{page_num}_{i}",
                    "text": chunk
                })
    else:
        print(f"⚠️ Tipo de archivo no soportado: {file_path.suffix}")

chunks_df = pd.DataFrame(records)
chunks_file = PROCESSED_DIR / "chunks.parquet"
chunks_df.to_parquet(chunks_file, index=False)

print(f"✅ Ingesta completada: {len(chunks_df)} chunks generados en {chunks_file}")
