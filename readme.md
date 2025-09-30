# README – Sistema RAG de Consulta Normativa UFRO

## 1. Descripción del Proyecto
Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** para consultas sobre normativa de la Universidad de La Frontera (UFRO).  
Permite hacer preguntas en lenguaje natural y obtener respuestas basadas en documentos normativos previamente indexados.

Se integra con **dos proveedores de LLM**:
- **ChatGPT** (OpenAI)
- **DeepSeek**

El sistema incluye:
- Indexación de documentos en **FAISS**.
- Generación de **embeddings** con `sentence-transformers`.
- Pipeline RAG con recuperación de documentos y síntesis de respuestas.
- CLI interactiva y API web con **Flask**.

---

## 2. Instalación

### Requisitos
Python 3.10+, con las siguientes dependencias:

```txt
pypdf
faiss-cpu
sentence-transformers
pandas
pyarrow
tqdm
openai
python-dotenv
matplotlib
scikit-learn
jupyter
ipykernel
flask
```

### Setup
1. Crear entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Crear archivo `.env` con tus keys:
```
OPENAI_API_KEY=tu_openai_api_key
DEEPSEEK_API_KEY=tu_deepseek_api_key
```

4. Descargar/colocar los documentos normativos en `data/raw/`.

5. Ingestar documentos y generar chunks + embeddings + índice FAISS:
```bash
python ingest.py
python embed.py
```

---

## 3. Uso

### CLI interactiva
```bash
python app.py --provider chatgpt
python app.py --provider deepseek
```

### Web (Flask)
Levantar servidor:
```bash
python web.py
```
Luego acceder desde el navegador:
```
http://<IP_PUBLICA>:5000
```
Se puede seleccionar el proveedor (`chatgpt` o `deepseek`) desde la interfaz web o mediante parámetros.

### Batch con gold set
Se puede evaluar con un conjunto de preguntas/respuestas (`eval/gold_set.jsonl`) usando `evaluate.py`.

---

## 4. Política de Ética y Abstención

- **Abstención**: Si la información solicitada no se encuentra en las fuentes normativas, el sistema responde exactamente:  
```
No encontré normativa relacionada
```

- **Privacidad**: No se almacenan datos de los usuarios fuera de la sesión activa.

- **Vigencia normativa**: Cada documento tiene su fecha de vigencia, que se incluye en la metadata (`sources.csv`).

---

## 5. Trazabilidad de Documentos

| doc_id     | Título                              | Fuente/URL                     | Tipo     | Vigencia         |
|----------- |----------------------------------- |--------------------------------|--------- |----------------|
| doc_001    | Reglamento de Convivencia UFRO      | https://www.ufro.cl            | pdf      | 01-01-2024     |
| doc_002    | Reglamento de Régimen de Estudios   | https://www.ufro.cl            | pdf      | 01-03-2023     |
| doc_003    | Calendario Académico 2025           | https://www.ufro.cl            | pdf      | 01-01-2025     |

> Cada `doc_id` se corresponde con los chunks generados, indexados en FAISS y referenciados en las respuestas RAG.

---

## 6. Evaluación y Métricas

- **H7 – Métricas de costo/latencia**:  
  - Medición de latencia end-to-end y por etapa (recuperación / LLM)  
  - Estimación de costo por consulta (tokens × tarifa pública)  
  - Guardado en CSV (`evaluation_results.csv`)

- **H7.5 – Comparativa y análisis**:  
  - Tabla/figura comparativa ChatGPT vs DeepSeek  
  - Trade-offs de similitud semántica y exactitud

- **H8 – Robustez**:  
  - Casos fuera de dominio → activar abstención  
  - Pruebas de alucinación → penalizar respuestas sin evidencia

