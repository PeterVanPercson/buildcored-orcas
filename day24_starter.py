"""
BUILDCORED ORCAS — Day 24: HardwareTA
Local RAG agent for hardware datasheet Q&A.
Index PDFs. Ask questions. Get cited answers.

Hardware concept: Datasheet RAG Pipeline
Engineers spend hours reading datasheets.
This automates retrieval — same pattern used in
AI-assisted hardware design tools (e.g. Flux.ai).

YOUR TASK:
1. Tune chunk size for datasheet content (TODO #1)
2. Ask a cross-document question (TODO #2)
3. Run: python day24_starter.py

PREREQS:
pip install PyMuPDF chromadb sentence-transformers
ollama running + ollama pull qwen2.5:3b
"""

import os
import sys
import subprocess
import time
import textwrap

# ============================================================
# CHECK DEPENDENCIES
# ============================================================

missing = []
try:
    import fitz  # PyMuPDF
except ImportError:
    missing.append("PyMuPDF")

try:
    import chromadb
except ImportError:
    missing.append("chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    missing.append("sentence-transformers")

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print(f"Fix: pip install {' '.join(missing)}")
    sys.exit(1)

import fitz
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    try:
        r = subprocess.run(["ollama", "list"],
                          capture_output=True, text=True, timeout=5)
        if "qwen2.5" not in r.stdout.lower():
            print("ERROR: Run: ollama pull qwen2.5:3b")
            sys.exit(1)
        print("✓ ollama ready")
    except:
        print("ERROR: ollama not running. Run: ollama serve")
        sys.exit(1)

check_ollama()
MODEL = "qwen2.5:3b"


# ============================================================
# DATASHEET SETUP
# ============================================================

DATASHEET_DIR = "datasheets"
os.makedirs(DATASHEET_DIR, exist_ok=True)


def get_sample_datasheets():
    """
    Check for PDFs in the datasheets folder.
    If none found, guide the student to get some.
    """
    pdfs = [f for f in os.listdir(DATASHEET_DIR)
            if f.lower().endswith(".pdf")]

    if pdfs:
        print(f"✓ Found {len(pdfs)} datasheet(s): {', '.join(pdfs)}")
        return [os.path.join(DATASHEET_DIR, p) for p in pdfs]

    print("\n⚠️  No datasheets found in ./datasheets/")
    print()
    print("Add datasheet PDFs to get started. Free sources:")
    print("  ESP32:   https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf")
    print("  STM32:   https://www.st.com/resource/en/datasheet/stm32f103c8.pdf")
    print("  DHT22:   search 'DHT22 AM2302 datasheet pdf'")
    print("  MPU6050: https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf")
    print()
    print("Place any PDF in the ./datasheets/ folder and re-run.")
    print()

    # Create a minimal synthetic datasheet for demo purposes
    print("Creating a synthetic demo datasheet for demonstration...")
    synthetic_path = create_synthetic_datasheet()
    return [synthetic_path]


def create_synthetic_datasheet():
    """Create a synthetic component datasheet PDF for demo."""
    doc = fitz.open()
    content_pages = [
        ("ORCAS-MCU32 Microcontroller Datasheet", """
ORCAS-MCU32 Overview
The ORCAS-MCU32 is a 32-bit microcontroller featuring dual-core Xtensa LX6
processors running at up to 240 MHz. It includes 520 KB of SRAM and 4 MB of
flash memory. The chip supports WiFi 802.11 b/g/n and Bluetooth 4.2.

Power Supply: 2.3V to 3.6V operating voltage.
Operating Temperature: -40°C to 85°C.
        """),
        ("GPIO and PWM", """
GPIO Configuration
The ORCAS-MCU32 features 34 programmable GPIO pins. GPIO pins 6-11 are
reserved for internal flash. Maximum current per GPIO: 40 mA.

PWM (Pulse Width Modulation)
16 independent PWM channels. PWM frequency range: 1 Hz to 40 MHz.
Resolution: up to 16-bit duty cycle. Ideal for LED dimming and servo control.
Timer groups: 2 hardware timer groups, each with 2 64-bit timers.
        """),
        ("I2C and SPI Interfaces", """
I2C Interface
Two I2C controllers. Standard mode: 100 kHz. Fast mode: 400 kHz.
7-bit and 10-bit addressing supported. ACK/NACK signaling per I2C spec.
Pull-up resistors: external 4.7k ohm recommended for 3.3V operation.

SPI Interface
Four SPI controllers (SPI0, SPI1 for flash, SPI2 and SPI3 for user).
Maximum clock: 80 MHz. Supports full-duplex. DMA support available.
        """),
        ("ADC Specifications", """
Analog-to-Digital Converter
Two SAR ADC modules, 18 channels total.
Resolution: 12-bit (0-4095). Reference voltage: 1.1V internal.
Input voltage range: 0-3.3V (with attenuation settings).
Attenuation options: 0dB (0-1.1V), 6dB (0-2.2V), 11dB (0-3.3V).
Sampling rate: up to 100K samples/second.
        """),
    ]

    for title, content in content_pages:
        page = doc.new_page()
        page.insert_text((50, 50), title,
                         fontsize=16, color=(0, 0, 0))
        page.insert_text((50, 90), content.strip(),
                         fontsize=10, color=(0, 0, 0))

    path = os.path.join(DATASHEET_DIR, "ORCAS-MCU32-datasheet.pdf")
    doc.save(path)
    doc.close()
    print(f"✓ Created synthetic datasheet: {path}")
    return path


# ============================================================
# TODO #1: Chunk size
# ============================================================
# Chunking splits long PDF text into pieces for embedding.
# Each chunk becomes one searchable unit in the vector DB.
#
# Too small (100 chars): misses context, answers lack detail
# Too large (2000 chars): embeds too broadly, retrieval is vague
# Sweet spot for datasheets: 400-600 characters
#
# Datasheets have tables, register maps, and electrical specs —
# these often need larger chunks to stay coherent.
#
CHUNK_SIZE = 500       # <-- Adjust this
CHUNK_OVERLAP = 100    # How many chars to repeat between chunks


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": os.path.basename(pdf_path)
            })
    doc.close()
    return pages


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split page text into overlapping chunks."""
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:  # Skip tiny fragments
                chunks.append({
                    "text": chunk_text,
                    "page": page_data["page"],
                    "source": page_data["source"],
                    "id": f"{page_data['source']}_p{page_data['page']}_c{start}"
                })
            start += chunk_size - overlap
    return chunks


# ============================================================
# VECTOR DATABASE SETUP
# ============================================================

def setup_chromadb():
    """Initialize chromadb with a persistent local collection."""
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete and recreate to ensure fresh index
    try:
        client.delete_collection("datasheets")
    except Exception:
        pass

    collection = client.create_collection(
        name="datasheets",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection


def index_chunks(collection, chunks, embedder):
    """Embed and store chunks in chromadb."""
    print(f"  Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]

    # Batch embedding (more efficient)
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = embedder.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())

    collection.add(
        documents=texts,
        embeddings=all_embeddings,
        ids=[c["id"] for c in chunks],
        metadatas=[{
            "page": c["page"],
            "source": c["source"]
        } for c in chunks]
    )


def retrieve(collection, embedder, query, n_results=3):
    """Find the most relevant chunks for a query."""
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results


# ============================================================
# LLM ANSWER GENERATION
# ============================================================

def generate_answer(query, retrieved_results):
    """Build context from retrieved chunks and ask the LLM."""
    docs = retrieved_results["documents"][0]
    metas = retrieved_results["metadatas"][0]

    # Build context with citations
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        context_parts.append(
            f"[Source {i+1}: {meta['source']}, Page {meta['page']}]\n{doc}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a hardware engineering assistant. Answer the question using ONLY the provided datasheet excerpts.
Cite which source and page number supports your answer.

DATASHEET EXCERPTS:
{context}

QUESTION: {query}

Answer concisely with specific values/specs where available. Always cite [Source N, Page X]:"""

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout.strip(), metas
    except subprocess.TimeoutExpired:
        return "[LLM timed out]", metas
    except Exception as e:
        return f"[Error: {e}]", metas


# ============================================================
# TODO #2: Cross-document question
# ============================================================
# Once you have multiple datasheets indexed, try a question
# that requires information from more than one document.
#
# Examples:
# "Which component has a higher maximum operating temperature,
#  the ESP32 or the STM32?"
#
# "What is the I2C pull-up resistor recommendation for the
#  MPU6050, and does the ESP32 have built-in pull-ups?"
#
# "Compare the ADC resolution of the ESP32 and STM32."
#
# These cross-document queries test whether your RAG is
# actually retrieving from multiple sources correctly.


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 55)
    print("  📚 HardwareTA — Day 24")
    print("=" * 55)
    print()

    # Load datasheets
    pdf_paths = get_sample_datasheets()

    # Load embedding model
    print("\nLoading embedding model (first run ~90 MB download)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Embedding model ready")

    # Extract and chunk
    print("\nProcessing datasheets...")
    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"  📄 {os.path.basename(pdf_path)}")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        print(f"     {len(pages)} pages → {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")

    # Index in chromadb
    print("\nIndexing in ChromaDB...")
    _, collection = setup_chromadb()
    index_chunks(collection, all_chunks, embedder)
    print(f"✓ Indexed {len(all_chunks)} chunks")

    # Interactive Q&A loop
    print("\n" + "=" * 55)
    print("  HardwareTA is ready. Ask hardware questions!")
    print("  Type 'quit' to exit.")
    print("=" * 55)
    print()
    print("  Try: 'What is the maximum GPIO current?'")
    print("       'What PWM frequencies are supported?'")
    print("       'What is the ADC resolution?'")
    print("       'What I2C modes are available?'")
    print()

    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        print("\n🔍 Retrieving relevant passages...")
        start = time.time()
        results = retrieve(collection, embedder, query)
        retrieve_time = time.time() - start

        print("💭 Generating answer...")
        answer_start = time.time()
        answer, metas = generate_answer(query, results)
        answer_time = time.time() - answer_start

        print()
        print("─" * 55)
        print(f"A: {answer}")
        print()
        print("  📎 Retrieved from:")
        for meta in metas:
            print(f"     • {meta['source']}, Page {meta['page']}")
        print(f"  ⚡ Retrieve: {retrieve_time:.2f}s | Generate: {answer_time:.1f}s")
        print("─" * 55)
        print()

    print("\nHardwareTA ended. See you tomorrow for Day 25!")


if __name__ == "__main__":
    main()
