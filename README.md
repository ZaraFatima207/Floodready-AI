# 🌊 FloodReady AI - Pakistan Disaster Preparedness Engine

FloodReady AI is a hyper-personalized disaster response and climate preparedness engine built specifically for Pakistan's flood-prone regions. Rather than returning generic safety guidelines, it uses Retrieval-Augmented Generation (RAG) powered by **Groq** to transform official disaster advisories (NDMA, PDMAs, FFD) into localized, timed 24-hour preparedness and evacuation plans in both **English and Roman Urdu**.

---

## 🚀 Key Features

- **Personalized Household Context:** Tailors actions based on district location, proximity to streams (e.g., Nullah Lai, Swat River), number of vulnerable members (elderly, infants), and vehicle availability.
- **Dual-Language Guidance:** Delivers step-by-step actions in English and conversational Roman Urdu.
- **Fast Groq LLM Inference:** Powered by Meta's open-weights models (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`) running on Groq's high-speed LPU infrastructure.
- **Lightweight CPU Vector RAG:** Uses `sentence-transformers/all-MiniLM-L6-v2` and `FAISS` for fast, zero-cost vector search without external embedding API dependencies.
- **Verified Emergency Directory:** One-stop offline-friendly directory for Rescue 1122, NDMA, PDMAs, Edhi, AlKhidmat, PRCS, and GBDMA.

---

## 🛠️ Local Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Free Groq API Key from [console.groq.com](https://console.groq.com/)

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone [https://github.com/your-username/floodready-ai.git](https://github.com/your-username/floodready-ai.git)
cd floodready-ai

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
