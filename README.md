# 🌊 FloodReady AI - Pakistan Disaster & Weather Intelligence Engine

FloodReady AI is an open-source, hyper-personalized disaster response and climate preparedness engine engineered specifically for Pakistan’s flood-prone regions. Built to address severe climate threats—ranging from urban flash floods in Rawalpindi’s Nullah Lai basin to Glacial Lake Outburst Floods (GLOFs) in Gilgit-Baltistan and riverine inundations across KPK, Punjab, and Sindh—FloodReady AI bridges the critical gap between official advisories and household-level action.

---

## 🚀 Key Features

- **Personalized Household Context:** Tailors actions based on district location, proximity to streams (e.g., Nullah Lai, Swat River), number of vulnerable members (elderly, infants), and vehicle availability.
- **Live Weather & Rain Prediction Analytics:** Interactive 7-day and 24-hour weather charts powered by Open-Meteo and Plotly.
- **Dual-Language Guidance:** Delivers step-by-step actions in English and conversational Roman Urdu.
- **Active Groq LLM Inference:** Powered by Meta's active open-weights models (`llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`) running on Groq's high-speed LPU infrastructure.
- **Lightweight CPU Vector RAG:** Uses `sentence-transformers/all-MiniLM-L6-v2` and `FAISS` for fast, zero-cost vector search without external embedding API dependencies.
- **Verified Disaster Directory:** Includes a direct directory connecting users to Rescue 1122, NDMA, PDMAs, Edhi Foundation, AlKhidmat, PRCS, and GBDMA control rooms.

---

## 🛠️ Local Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Free Groq API Key from [console.groq.com](https://console.groq.com/)

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone [https://github.com/ZaraFatima207/Floodready-AI.git](https://github.com/ZaraFatima207/Floodready-AI.git)
cd Floodready-AI

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
