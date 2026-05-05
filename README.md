# MedRAG: IoT-Driven Health Monitoring with Advanced RAG and Small Language Models

> **MAI556: Generative AI — Course Project | Alfaisal University, Spring 2026**

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Installation Guide](#installation-guide)
- [Dependencies](#dependencies)
- [Setup and Usage Instructions](#setup-and-usage-instructions)
- [Example Usage and Demo](#example-usage-and-demo)
- [Dataset Information](#dataset-information)
- [Code Structure](#code-structure)
- [Team](#team)

---

## Project Overview

**MedRAG** is an Advanced Retrieval-Augmented Generation (RAG) system for IoT-driven clinical decision support. It simulates three medical IoT devices (glucometer, blood pressure monitor, heart rate sensor), converts raw sensor readings into structured clinical queries, retrieves relevant evidence from a curated medical knowledge base using a hybrid search pipeline, and generates grounded medical explanations using a Small Language Model (SLM).

### Key Features
- **IoT Simulation** — realistic sensor readings across normal, warning, and critical severity levels
- **Rule-Based Query Rewriting** — converts raw numbers into natural-language clinical questions
- **Hybrid Retrieval** — combines FAISS semantic search + BM25 keyword search via Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Re-Ranking** — refines retrieval results for higher relevance precision
- **SLM Generation** — grounded medical explanations using Qwen2.5-1.5B-Instruct
- **Interactive UI** — Gradio interface with sliders and preset scenarios
- **RAGAS Evaluation** — automated evaluation of Faithfulness, Answer Relevancy, and Context Precision

---

## System Architecture

```
IoT Sensor Readings  (Block 5)
        ↓
Rule-Based Query Rewriter  (Block 10)
        ↓
┌─────────────────────────────────┐
│         Hybrid Search           │
│  FAISS Semantic  +  BM25 Keyword│  (Blocks 6, 7, 8)
│         ↓ RRF Fusion            │
└─────────────────────────────────┘
        ↓
Cross-Encoder Re-Ranking  (Block 9)
        ↓
SLM Response Generation  (Block 11)
   Qwen2.5-1.5B-Instruct
        ↓
Gradio UI  (Block 12)  +  RAGAS Evaluation  (Block 13)
```

---

## Installation Guide

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Go to **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**
3. Open the notebook file `MedRAG_Pipeline.ipynb`
4. Run **Block 2** to install all dependencies automatically:

```python
!pip install -q transformers sentence-transformers accelerate
!pip install -q faiss-cpu rank-bm25
!pip install -q langchain-text-splitters
!pip install -q gradio
!pip install -q ragas datasets langchain-huggingface
```

> ⚠️ After installation, Colab may prompt you to **restart the runtime** — click "Restart Runtime" and continue from Block 3.

### Option B — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/your-username/medrag.git
cd medrag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dependencies

All dependencies are listed in `requirements.txt`. Core libraries:

| Library | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Deep learning framework |
| `transformers` | ≥ 4.40 | HuggingFace model loading |
| `sentence-transformers` | ≥ 2.7 | Embedding + Cross-encoder |
| `faiss-cpu` | ≥ 1.7 | Vector similarity search |
| `rank-bm25` | ≥ 0.2.2 | BM25 keyword retrieval |
| `langchain-text-splitters` | ≥ 0.2 | Document chunking |
| `gradio` | ≥ 4.0 | Interactive UI |
| `ragas` | ≥ 0.1 | RAG evaluation framework |
| `datasets` | ≥ 2.0 | HuggingFace datasets |
| `langchain-huggingface` | ≥ 0.1 | LangChain + HuggingFace bridge |

> **GPU requirement:** NVIDIA GPU with ≥ 8 GB VRAM recommended (tested on Tesla T4, 15 GB). CPU-only is possible but significantly slower.

---

## Setup and Usage Instructions

### Step 1 — Verify Environment (Block 1)
Run Block 1 to confirm GPU availability:
```
✅ Ready to proceed! GPU: Tesla T4 (15.0 GB)
```
If you see ⚠️, enable GPU first (see Installation Guide).

### Step 2 — Install Libraries (Block 2)
Run Block 2. Takes ~2-3 minutes on first run.

### Step 3 — Build Knowledge Base (Block 3)
Constructs the medical knowledge base from four curated documents (ADA, WHO, JNC 8 guidelines). No internet access required — content is embedded in code.

```
✓ Created: diabetes_glucose.txt
✓ Created: hypertension_bp.txt
✓ Created: cardiovascular.txt
✓ Created: emergency_thresholds.txt
```

### Step 4 — Chunk Documents (Block 4)
Splits documents into 24 searchable chunks (~420 chars each).

### Step 5 — Build Retrieval Indexes (Blocks 6 & 7)
- **Block 6:** Encodes all chunks into 384-dim vectors → FAISS index
- **Block 7:** Tokenizes chunks → BM25 index

### Step 6 — Run Full Pipeline (Blocks 8–11)
Blocks 8–11 build the Hybrid Search, Re-Ranker, Query Rewriter, and SLM Generator. Run sequentially.

### Step 7 — Launch UI (Block 12)
```python
demo.launch(share=True)
# → Running on public URL: https://xxxxx.gradio.live
```
Open the URL in any browser. The UI is usable on mobile.

### Step 8 — Evaluate (Block 13)
Runs RAGAS evaluation on 6 test cases and prints Faithfulness, Answer Relevancy, and Context Precision scores.

---

## Example Usage and Demo

### Quick Demo via UI

1. Launch Block 12 → open the Gradio URL
2. Click **🔴 Critical Case** preset button
3. Click **🔍 Analyze Readings**
4. Wait ~10 seconds for the SLM to generate
5. View the medical explanation + open the "Show Sources & Evidence" accordion

### Quick Demo via Code

```python
from medrag import generate_iot_reading, run_full_pipeline

# Generate a critical IoT reading for a 58-year-old patient
reading = generate_iot_reading(severity="critical", patient_age=58)
print(reading)
# Glucose: 387.2 mg/dL | BP: 195/118 mmHg | HR: 152 bpm | Age: 58 | Severity: CRITICAL

# Run the full Advanced RAG pipeline
result = run_full_pipeline(reading)

print(result["overall_concern"])   # CRITICAL
print(result["response"])          # Full medical explanation
print(result["sources"])           # ['emergency_thresholds.txt', 'diabetes_glucose.txt']
```

### Severity Presets Reference

| Preset | Glucose (mg/dL) | BP (mmHg) | HR (bpm) |
|---|---|---|---|
| 🟢 Normal | 88 | 117/76 | 78 |
| 🟡 Warning | 165 | 145/92 | 108 |
| 🔴 Critical | 380 | 195/118 | 145 |

### Sample Output — Critical Case

```
📋 Medical Classifications:
   🔴 Glucose:        severe hyperglycemia
   🔴 Blood Pressure: stage 2 hypertension
   🔴 Heart Rate:     severe tachycardia

⚠️  Overall Concern: CRITICAL

💬 Medical Explanation:
Based on the readings provided, this patient is experiencing 
several critical conditions requiring immediate attention...

📚 Sources Used:
   • emergency_thresholds.txt
   • diabetes_glucose.txt
```

---

## Dataset Information

### Medical Knowledge Base (built-in)

The knowledge base is embedded directly in the code (Block 3) and requires no downloads. It is adapted from the following public-domain sources:

| Document | Source | Coverage |
|---|---|---|
| `diabetes_glucose.txt` | ADA Standards of Care 2024 | Glucose ranges, DKA, hypoglycemia |
| `hypertension_bp.txt` | JNC 8 Guidelines (James et al., 2014) | BP categories, treatment |
| `cardiovascular.txt` | WHO HEARTS Package 2018 | Heart rate, arrhythmia, CVD |
| `emergency_thresholds.txt` | WHO Emergency Care Guidelines 2018 | Critical thresholds, first aid |

> All documents are adapted from **public-domain** clinical guidelines and are free to use for research and educational purposes.

### No External Dataset Required

This project does not use a patient dataset. IoT readings are synthetically generated by the simulation layer (Block 5) using clinically validated ranges.

---

## Code Structure

```
MedRAG_Pipeline.ipynb          ← Main Colab notebook (all 13 blocks)
requirements.txt               ← Python dependencies
README.md                      ← This file
knowledge_base/                ← Auto-created by Block 3
   diabetes_glucose.txt
   hypertension_bp.txt
   cardiovascular.txt
   emergency_thresholds.txt
```

### Block Summary

| Block | Description |
|---|---|
| Block 1 | Environment verification (Python, PyTorch, GPU) |
| Block 2 | Library installation |
| Block 3 | Medical knowledge base construction |
| Block 4 | Document chunking (RecursiveCharacterTextSplitter) |
| Block 5 | IoT sensor simulation (3 devices × 3 severity levels) |
| Block 6 | Semantic search — FAISS + MiniLM embeddings |
| Block 7 | Keyword search — BM25 index |
| Block 8 | Hybrid search — Reciprocal Rank Fusion (RRF) |
| Block 9 | Cross-encoder re-ranking |
| Block 10 | Rule-based query rewriting |
| Block 11 | SLM response generation (Qwen2.5-1.5B-Instruct) |
| Block 12 | Gradio interactive UI |
| Block 13 | RAGAS evaluation (Faithfulness, Relevancy, Precision) |

---

**Institution:** Alfaisal University — College of Engineering and Advanced Computing
**Course:** MAI556: Generative AI | Spring 2026

---

> *This project was developed as part of the MAI556 course project requirement. The medical explanations generated are for educational purposes only and should not be used as a substitute for professional medical advice.*
