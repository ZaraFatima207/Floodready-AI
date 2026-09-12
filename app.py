import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodReady AI - Pakistan National Disaster Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.3rem; color: #1E3A8A; font-weight: 800; margin-bottom: 2px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .alert-card { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .info-card { background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .guide-card { background-color: #F0FDF4; border-left: 5px solid #22C55E; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .stButton>button { width: 100%; background-color: #1E3A8A; color: white; font-weight: bold; border-radius: 8px; height: 3.2em; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. HARDCODED / SECRETS GROQ API KEY CONFIGURATION
# ---------------------------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

FAST_NON_LLAMA_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini"
]

# ---------------------------------------------------------------------------
# 3. SYSTEM PROMPT DEFINITION
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are FloodReady AI, a hyper-personalized disaster response and preparedness engine operating in Pakistan.
Your task is to transform raw disaster management advisories, weather telemetry, and geographic hazard data into structured, timed, and actionable 24-hour preparedness and evacuation plans tailored to specific Pakistani households and locations.

### INPUT CONTEXT
- PROVINCE: {province}
- DISTRICT & REGION: {location}
- EXACT ADDRESS / NEIGHBORHOOD: {address}
- HOUSEHOLD PROFILE: {household_profile}
- RETRIEVED ADVISORIES & EMERGENCY DATA:
{context}

### OPERATIONAL DIRECTIVES & GUARDRAILS
1. STRICT ADHERENCE TO RETRIEVED DATA:
   - Rely strictly on verified Pakistan emergency contacts (Rescue 1122, NDMA, PDMA, GBDMA, SDMA) and local shelter protocols.
   - Do NOT invent shelter addresses or phone numbers. If local shelter information is missing, direct the user to Rescue 1122 or the District Administration / Deputy Commissioner's Office.
2. DEMOGRAPHIC PERSONALIZATION:
   - Large Family / Group Scaling: Tailor instructions for household capacity up to {capacity} members. For large households, assign leadership roles, split evacuation duties, and organize buddy systems.
   - Vulnerable Members (Elderly / Infants / Special Needs): Explicitly prioritize medical kits, mobility aids, infant formula, and early evacuation protocols before general tasks.
   - Vehicle Protocols: Provide concrete instructions for parking or securing motor vehicles on elevated ground away from drainage channels (e.g., Nullah Lai, Lyari River, Swat Riverbed, Indus riverine bunds).
3. DUAL-LANGUAGE DELIVERY:
   - Present every main action item in clear English, followed immediately by conversational Roman Urdu (Urdu in Latin script) to ensure accessibility across all literacy levels.
4. REGIONAL HAZARD ADAPTATION:
   - High-Altitude / Mountainous (GB, KPK, AJK): Emphasize GLOF acoustic warnings, high-ground foot migration, landslide hazards, and ridge safety.
   - Plain & Riverine Basins (Punjab, Sindh, Balochistan): Emphasize electrical main breaker shutdown, drain spillover containment, barrage discharge timelines, water purification, and snakebite precaution.

### OUTPUT FORMAT REQUIREMENTS
Format your response cleanly in Markdown:
1. ⚠️ LIVE THREAT & HAZARD ASSESSMENT ({location}, {province})
2. 📋 PERSONALIZED 24-HOUR PREPAREDNESS PLAN
   - Phase 1: Pre-Flood Preparation (T-24 Hours) [English & Roman Urdu]
   - Phase 2: Mandatory Evacuation Protocol (On Advisory / Level 2 Alert) [English & Roman Urdu]
3. 📞 VERIFIED LOCAL & PROVINCIAL EMERGENCY HELPLINES
"""

# ---------------------------------------------------------------------------
# 4. PAKISTAN GEOGRAPHIC HIERARCHY & COORDINATES DATASET
# ---------------------------------------------------------------------------
PAKISTAN_GEOGRAPHY = {
    "Punjab": {
        "Rawalpindi": {"regions": ["Nullah Lai Corridor", "Gawalmandi", "Arya Mohallah", "Westridge", "Sadiqabad", "General/Other"], "lat": 33.5970, "lon": 73.0439},
        "Lahore": {"regions": ["Ravi Riverbed", "Shahdara", "Johar Town Drains", "Gulberg", "General/Other"], "lat": 31.5204, "lon": 74.3587},
        "D.G. Khan": {"regions": ["Hill Torrent Corridor (Kaha/Mithawan)", "Taunsa Sharif", "City Area", "General/Other"], "lat": 30.0561, "lon": 70.6348},
        "Rajanpur": {"regions": ["Rajanpur City", "Jampur", "Mithankot (Indus Confluence)", "General/Other"], "lat": 29.1035, "lon": 70.3250},
        "Multan": {"regions": ["Chenab Riverbank", "Shershah Bund", "City Center", "General/Other"], "lat": 30.1575, "lon": 71.5249},
        "Sialkot": {"regions": ["Aik Nullah Basin", "Bhed Nullah Zone", "City Center", "General/Other"], "lat": 32.4945, "lon": 74.5229},
        "Faisalabad": {"regions": ["Paharang Drain Zone", "Samundri", "Jaranwala", "General/Other"], "lat": 31.4504, "lon": 73.1350},
        "Rahim Yar Khan": {"regions": ["Indus Riverine Belt", "Sadiqabad", "Khanpur", "General/Other"], "lat": 28.4212, "lon": 70.2989}
    },
    "Sindh": {
        "Sukkur": {"regions": ["Sukkur Barrage Upstream", "Indus Bund", "City Area", "General/Other"], "lat": 27.7052, "lon": 68.8574},
        "Karachi": {"regions": ["Lyari River Basin", "Malir River Bed", "Surjani Town", "DHA / Clifton", "Orangi Nullah", "General/Other"], "lat": 24.8607, "lon": 67.0011},
        "Larkana": {"regions": ["Rice Canal Belt", "Mohenjo-daro Buffer", "City Area", "General/Other"], "lat": 27.5598, "lon": 68.2120},
        "Dadu": {"regions": ["MNV Drain / Manchhar Lake", "Mehar", "Johi Ring Bund", "General/Other"], "lat": 26.8913, "lon": 67.7788},
        "Hyderabad": {"regions": ["Phuleli Canal Corridor", "Latifabad", "Qasimabad", "General/Other"], "lat": 25.3960, "lon": 68.3578},
        "Thatta": {"regions": ["Keti Bandar Coastal", "Sujawal Bridge Zone", "General/Other"], "lat": 24.7475, "lon": 67.9239},
        "Badin": {"regions": ["LBOD Coastal Drain Zone", "Matli", "Tando Bago", "General/Other"], "lat": 24.6560, "lon": 68.8370},
        "Jacobabad": {"regions": ["Thul Canal Belt", "City Ring Bund", "General/Other"], "lat": 28.2810, "lon": 68.4376}
    },
    "Khyber Pakhtunkhwa (KPK)": {
        "Swat": {"regions": ["Mingora Riverbank", "Bahrain", "Kalam Valley", "Khwazakhela", "General/Other"], "lat": 34.7717, "lon": 72.3602},
        "Nowshera": {"regions": ["Kabul River Basin", "Nowshera Cantt Lowlands", "Pabbi", "General/Other"], "lat": 34.0153, "lon": 71.9747},
        "Peshawar": {"regions": ["Bara River Corridor", "Warsak Road Zone", "City Center", "General/Other"], "lat": 34.0151, "lon": 71.5249},
        "Charsadda": {"regions": ["Jindi River Belt", "Khyali River Zone", "Shabqadar", "General/Other"], "lat": 34.1482, "lon": 71.7406},
        "Chitral": {"regions": ["Chitral River Valley", "Reshun GLOF Zone", "Bumburet Valley", "General/Other"], "lat": 35.8510, "lon": 71.7869},
        "D.I. Khan": {"regions": ["Daraban Hill Torrent Corridor", "Proboa Bund", "City Zone", "General/Other"], "lat": 31.8314, "lon": 70.9019},
        "Abbottabad": {"regions": ["Dor River Basin", "Havelian", "City Ravine Drains", "General/Other"], "lat": 34.1688, "lon": 73.2215}
    },
    "Balochistan": {
        "Quetta": {"regions": ["Sariab Nullah", "Hanna Valley Ravine", "City Bowl", "General/Other"], "lat": 30.1798, "lon": 66.9750},
        "Jaffarabad": {"regions": ["Dera Allah Yar", "Usta Muhammad Canal Belt", "General/Other"], "lat": 28.4323, "lon": 68.0412},
        "Nasirabad": {"regions": ["Dera Murad Jamali", "Pat Feeder Canal", "General/Other"], "lat": 28.5463, "lon": 68.2231},
        "Gwadar": {"regions": ["Old Town Lowlands", "Expressway Runoff Channel", "Pasni Coastal", "General/Other"], "lat": 25.1264, "lon": 62.3225},
        "Khuzdar": {"regions": ["Wadh Torrent Zone", "Nal River Corridor", "General/Other"], "lat": 27.8165, "lon": 66.6057},
        "Lasbela": {"regions": ["Hub River Corridor", "Uthal Lowlands", "Bela Town", "General/Other"], "lat": 26.2269, "lon": 66.3138}
    },
    "Gilgit-Baltistan (GB)": {
        "Hunza": {"regions": ["Passu Glacier Stream", "Hassanabad GLOF Bridge Zone", "Aliabad", "General/Other"], "lat": 36.3167, "lon": 74.6500},
        "Gilgit": {"regions": ["Gilgit River Delta", "Jotial Nullah Corridor", "Danyore", "General/Other"], "lat": 35.9208, "lon": 74.3144},
        "Skardu": {"regions": ["Indus River Valley", "Satpara Stream Corridor", "General/Other"], "lat": 35.2971, "lon": 75.6333},
        "Ghizer": {"regions": ["Shisher Glacier Zone", "Gupis Valley", "Ishkoman", "General/Other"], "lat": 36.1736, "lon": 73.7667},
        "Diamer": {"regions": ["Chilas Indus Riverbed", "Babusar Nullah", "General/Other"], "lat": 35.4206, "lon": 74.0967}
    },
    "Azad Jammu & Kashmir (AJK)": {
        "Muzaffarabad": {"regions": ["Neelum & Jhelum Confluence", "Lower Chattar Lowlands", "General/Other"], "lat": 34.3700, "lon": 73.4711},
        "Mirpur": {"regions": ["Mangla Reservoir Buffer Zone", "Dadyal", "General/Other"], "lat": 33.1484, "lon": 73.7519},
        "Rawalakot": {"regions": ["Poonch River Valley", "Hajira", "General/Other"], "lat": 33.8584, "lon": 73.7653},
        "Neelum Valley": {"regions": ["Kutton Stream Corridor", "Sharda Riverbank", "General/Other"], "lat": 34.7933, "lon": 73.9114}
    },
    "Islamabad Capital Territory (ICT)": {
        "Islamabad": {"regions": ["Sector E-11 Ravine Drain", "Nullah Korang Corridor", "Rawal Dam Spillway Channel", "Sector F-6/F-7 Streams", "General/Other"], "lat": 33.6844, "lon": 73.0479}
    }
}

# ---------------------------------------------------------------------------
# 5. HISTORICAL PAKISTAN FLOOD ANALYTICS DATASETS
# ---------------------------------------------------------------------------
HISTORICAL_FLOOD_EVENTS = pd.DataFrame([
    {"Year": 2010, "Event": "2010 Super Floods", "Fatalities": 1985, "Affected_People_Millions": 20.0, "Economic_Loss_Billion_USD": 10.0, "Primary_Provinces": "KPK, Punjab, Sindh, Balochistan"},
    {"Year": 2012, "Event": "2012 Monsoon Surges", "Fatalities": 471, "Affected_People_Millions": 5.0, "Economic_Loss_Billion_USD": 2.5, "Primary_Provinces": "Sindh, Punjab, Balochistan"},
    {"Year": 2014, "Event": "2014 Chenab & Jhelum Floods", "Fatalities": 367, "Affected_People_Millions": 2.5, "Economic_Loss_Billion_USD": 1.8, "Primary_Provinces": "Punjab, AJK"},
    {"Year": 2020, "Event": "2020 Urban Deluge (Karachi)", "Fatalities": 410, "Affected_People_Millions": 2.2, "Economic_Loss_Billion_USD": 1.5, "Primary_Provinces": "Sindh, KPK, Balochistan"},
    {"Year": 2022, "Event": "2022 Catastrophic Monsoon", "Fatalities": 1739, "Affected_People_Millions": 33.0, "Economic_Loss_Billion_USD": 30.1, "Primary_Provinces": "Sindh, Balochistan, KPK, Punjab, GB"},
    {"Year": 2024, "Event": "2024 Hill Torrent & Flash Floods", "Fatalities": 320, "Affected_People_Millions": 3.8, "Economic_Loss_Billion_USD": 2.2, "Primary_Provinces": "KPK, Punjab, GB, Balochistan"}
])

PROVINCIAL_DAMAGE_2022 = pd.DataFrame([
    {"Province": "Sindh", "Affected_Millions": 14.5, "Houses_Damaged_Thousands": 1840, "Cropland_Flooded_M_Acres": 4.2},
    {"Province": "Balochistan", "Affected_Millions": 9.2, "Houses_Damaged_Thousands": 240, "Cropland_Flooded_M_Acres": 1.1},
    {"Province": "Punjab", "Affected_Millions": 4.8, "Houses_Damaged_Thousands": 85, "Cropland_Flooded_M_Acres": 1.0},
    {"Province": "KPK", "Affected_Millions": 4.3, "Houses_Damaged_Thousands": 92, "Cropland_Flooded_M_Acres": 0.5},
    {"Province": "GB & AJK", "Affected_Millions": 0.2, "Houses_Damaged_Thousands": 12, "Cropland_Flooded_M_Acres": 0.1}
])

# ---------------------------------------------------------------------------
# 6. LIVE WEATHER FETCHING (OPEN-METEO API)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_weather_forecast(lat: float, lon: float):
    """Fetches 7-day daily and 24-hour hourly weather forecast from Open-Meteo API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weathercode&hourly=temperature_2m,precipitation_probability,rain&timezone=Asia%2FKarachi"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.warning(f"Live weather telemetry unavailable: {e}")
    return None

def interpret_weather_code(code: int) -> str:
    """Maps WMO Weather Codes to human-readable text with icons."""
    codes = {
        0: "Clear Sky ☀️", 1: "Mainly Clear 🌤️", 2: "Partly Cloudy ⛅", 3: "Overcast ☁️",
        45: "Foggy 🌫️", 51: "Light Drizzle 🌧️", 61: "Slight Rain 🌧️",
        63: "Moderate Rain 🌧️", 65: "Heavy Downpour ⛈️", 80: "Rain Showers 🌦️",
        95: "Thunderstorm 🌩️", 96: "Severe Thunderstorm with Hail ⛈️"
    }
    return codes.get(code, "Cloudy / Variable ⛅")

# ---------------------------------------------------------------------------
# 7. KNOWLEDGE BASE & FAISS VECTOR STORE
# ---------------------------------------------------------------------------
PAKISTAN_DISASTER_KNOWLEDGE_BASE = [
    {
        "district": "Rawalpindi", "province": "Punjab", "authority": "PDMA Punjab / Rescue 1122",
        "content": "Rawalpindi Nullah Lai Alert Levels: 11 ft = Alert Level 1 (Monitoring); 14 ft = Alert Level 2 (Standby Evacuation); 18 ft = Emergency Evacuation. Relief Shelters: Govt Gordon College Rawalpindi, Govt High School Westridge, Govt College Asghar Mall. Helplines: Rescue 1122, Rawalpindi Control Room: 051-929296, PDMA Punjab: 1129."
    },
    {
        "district": "Swat", "province": "KPK", "authority": "PDMA KPK",
        "content": "Swat River High Risk Zones: Kalam, Bahrain, Madyan, Mingora riverbanks, Khwazakhela. Shelters: Govt High School Kalam, Govt Degree College Mingora. Helplines: Rescue 1122, PDMA KPK Emergency Hotline: 1700, WhatsApp: 0316-4261700. Foot migration to higher mountain ridges is mandatory upon upstream warning."
    },
    {
        "district": "Hunza", "province": "Gilgit-Baltistan", "authority": "GBDMA / AKAH",
        "content": "Hunza GLOF Hazard Zones: Hassanabad Bridge, Passu Glacier stream, Shishper Lake. Early warning sirens installed by AKAH. Evacuation Shelters: AKAH Community Centers, Govt Boys High School Hunza. Helplines: GBDMA Control Room: 05811-920830, Rescue 1122 GB."
    },
    {
        "district": "Sukkur", "province": "Sindh", "authority": "PDMA Sindh",
        "content": "Sukkur Barrage Flood Warning Thresholds: Medium Flood (400,000 cusecs), High Flood (500,000 cusecs), Super Flood (>700,000 cusecs). Emergency Shelters: Public Sector Colleges and PDMA Tent Cities. Helplines: PDMA Sindh Control Room: 021-99332005, Rescue 1122 Sindh, Edhi: 115."
    },
    {
        "district": "Quetta", "province": "Balochistan", "authority": "PDMA Balochistan",
        "content": "Quetta Hill Torrents & Ravine Floods: Sariab Nullah, Hanna Urak Valley. Shelters: Railway Community Center, Govt Science College Quetta. Helplines: PDMA Balochistan Control Room: 081-9241133, Rescue 1122 Balochistan."
    },
    {
        "district": "National", "province": "National Directory", "authority": "NDMA Pakistan",
        "content": "Pakistan National Emergency Directory: NDMA Hotline: 051-111-157-157 | Rescue 1122 (All Provinces) | Edhi Ambulance: 115 | Chhipa Welfare: 1020 | AlKhidmat Foundation: 1023 | Pakistan Red Crescent (PRCS): 1030 | Flood Forecasting Division (FFD): 042-99200139."
    }
]

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [
        Document(
            page_content=item["content"],
            metadata={"district": item["district"], "province": item["province"], "authority": item["authority"]}
        )
        for item in PAKISTAN_DISASTER_KNOWLEDGE_BASE
    ]
    return FAISS.from_documents(documents, embeddings)

# ---------------------------------------------------------------------------
# 8. SIDEBAR CONTROLS & GEOGRAPHIC INPUTS
# ---------------------------------------------------------------------------
st.sidebar.title("🌊 FloodReady AI")
st.sidebar.markdown("**National Pakistan Disaster Engine**")

# Model Selection Dropdown (Preserved Fast Non-Llama Options)
selected_model = st.sidebar.selectbox(
    "🤖 LLM Engine (Fast Non-Llama)",
    FAST_NON_LLAMA_MODELS,
    index=0,
    help="Select an active fast non-Llama model hosted on Groq."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 National Location Hierarchy")

# Province Selector
province_list = list(PAKISTAN_GEOGRAPHY.keys())
selected_province = st.sidebar.selectbox("Select Province / Territory", province_list, index=0)

# District Selector (Cascading)
district_map = PAKISTAN_GEOGRAPHY[selected_province]
district_list = list(district_map.keys())
selected_district = st.sidebar.selectbox("Select District", district_list, index=0)

# Region Selector (Cascading)
district_data = district_map[selected_district]
region_list = district_data["regions"]
selected_region = st.sidebar.selectbox("Select Specific Tehsil / Ravine / Zone", region_list, index=0)

# Specific Address Input
specific_address = st.sidebar.text_input(
    "Street / Mohallah / Landmark",
    value="House #12, Street 4, Near Local Mosque/Stream",
    help="Enter your exact street name or nearest landmark."
)

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Household Demographics")

# Household Capacity Input (Max increased to 100 as requested)
total_members = st.sidebar.number_input("Total Household Members", min_value=1, max_value=100, value=6)
elderly_count = st.sidebar.number_input("Elderly Members (60+ yrs)", min_value=0, max_value=50, value=1)
children_count = st.sidebar.number_input("Children / Infants", min_value=0, max_value=50, value=2)
special_needs_count = st.sidebar.number_input("Special Needs / Disabled Members", min_value=0, max_value=50, value=0)
has_vehicle = st.sidebar.checkbox("Household Has Motor Vehicle", value=True)

# ---------------------------------------------------------------------------
# 9. MAIN APP LAYOUT & TABS
# ---------------------------------------------------------------------------
st.markdown("<div class='main-header'>🌊 FloodReady AI Pakistan</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>National Early Warning, Personalized Preparedness Plans & Disaster Analytics</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 24-Hour AI Action Plan", 
    "📊 Weather & Historical Flood Radar", 
    "📞 Nationwide Directory", 
    "🌿 Ecosystem Protection",
    "📖 How To Use App Guide"
])

# ---------------------------------------------------------------------------
# TAB 1: 24-HOUR ACTION PLAN GENERATOR
# ---------------------------------------------------------------------------
with tab1:
    st.markdown(f"""
    <div class='alert-card'>
        <strong>⚠️ Selected Target Zone:</strong> {selected_region}, District {selected_district}, {selected_province}<br/>
        <strong>🏠 Household Capacity & Profile:</strong> {total_members} Members ({elderly_count} Elderly, {children_count} Children/Infants, {special_needs_count} Special Needs) | <strong>Vehicle Access:</strong> {'Yes' if has_vehicle else 'No'}<br/>
        <strong>📍 Address:</strong> {specific_address}
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Generate Personalized 24-Hour Action Plan"):
        if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
            st.error("⚠️ GROQ_API_KEY is not configured in code or Streamlit Secrets. Please set your key in `GROQ_API_KEY`.")
        else:
            with st.spinner("Analyzing hydrometeorological risk & constructing customized plan..."):
                try:
                    # 1. Retrieve Context from FAISS Vector Store
                    vector_store = load_vector_store()
                    query = f"Flood emergency evacuation shelter helpline {selected_province} {selected_district} {selected_region}"
                    retrieved_docs = vector_store.similarity_search(query, k=3)
                    
                    context_str = "\n\n".join([
                        f"[Authority: {doc.metadata['authority']} - {doc.metadata['province']}]\n{doc.page_content}"
                        for doc in retrieved_docs
                    ])
                    
                    household_summary = (
                        f"Total Capacity: {total_members} members | "
                        f"Elderly: {elderly_count} | "
                        f"Children/Infants: {children_count} | "
                        f"Special Needs: {special_needs_count} | "
                        f"Vehicle Access: {'Yes' if has_vehicle else 'No'}"
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", SYSTEM_PROMPT),
                        ("human", "Generate my personalized flood preparedness and evacuation plan.")
                    ])

                    # 2. Execution Loop with Fallback
                    models_to_try = [selected_model] + [m for m in FAST_NON_LLAMA_MODELS if m != selected_model]
                    response_text = None
                    last_error = None
                    successful_model = None

                    for model_name in models_to_try:
                        try:
                            llm = ChatGroq(
                                groq_api_key=GROQ_API_KEY,
                                model_name=model_name,
                                temperature=0.1
                            )
                            chain = prompt | llm | StrOutputParser()
                            response_text = chain.invoke({
                                "province": selected_province,
                                "location": f"{selected_district} ({selected_region})",
                                "address": specific_address,
                                "capacity": total_members,
                                "household_profile": household_summary,
                                "context": context_str
                            })
                            successful_model = model_name
                            break
                        except Exception as err:
                            last_error = err
                            continue

                    if response_text:
                        if successful_model != selected_model:
                            st.info(f"ℹ️ Selected model endpoint was unavailable. Plan successfully generated using fallback model: `{successful_model}`")
                        st.markdown("---")
                        st.markdown(response_text)
                    else:
                        st.error(f"Execution Error across all non-Llama models: {last_error}")
                    
                except Exception as e:
                    st.error(f"System Exception: {str(e)}")

# ---------------------------------------------------------------------------
# TAB 2: LIVE WEATHER & HISTORICAL FLOOD ANALYTICS
# ---------------------------------------------------------------------------
with tab2:
    st.subheader(f"🌦️ Real-Time Telemetry & Weather Radar for {selected_district}, {selected_province}")
    
    lat = district_data["lat"]
    lon = district_data["lon"]
    weather_data = fetch_weather_forecast(lat, lon)
    
    if weather_data and "daily" in weather_data:
        daily = weather_data["daily"]
        dates = daily["time"]
        max_temps = daily["temperature_2m_max"]
        min_temps = daily["temperature_2m_min"]
        precip_sum = daily["precipitation_sum"]
        precip_prob = daily["precipitation_probability_max"]
        weather_codes = daily["weathercode"]
        
        # Weather Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Condition", interpret_weather_code(weather_codes[0]))
        col2.metric("Today's Max Temp", f"{max_temps[0]} °C")
        col3.metric("7-Day Rain Total", f"{sum(precip_sum):.1f} mm")
        col4.metric("Peak Rain Probability", f"{max(precip_prob)} %")
        
        st.markdown("---")
        st.markdown("### 📊 7-Day Hydrometeorological Forecast")
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.12,
            subplot_titles=("Daily Temperature Range (°C)", "Expected Rainfall (mm) & Precipitation Probability (%)")
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=max_temps, name="Max Temp (°C)", line=dict(color="#EF4444", width=3), mode="lines+markers"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=min_temps, name="Min Temp (°C)", line=dict(color="#3B82F6", width=2, dash="dash"), mode="lines+markers"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=dates, y=precip_sum, name="Rainfall (mm)", marker_color="#1D4ED8", opacity=0.75),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=precip_prob, name="Rain Prob (%)", line=dict(color="#10B981", width=2.5), mode="lines+markers"),
            row=2, col=1
        )
        
        fig.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Weather data is currently updating. Please refresh.")

    st.markdown("---")
    st.subheader("🇵🇰 Historical Flood Impact & Damage Analytics (Pakistan)")
    st.markdown("Comparative historical data highlighting major flood occurrences, affected populations, economic damages, and provincial vulnerability.")

    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        st.markdown("#### Historical Flood Losses (2010 - 2024)")
        fig_events = px.bar(
            HISTORICAL_FLOOD_EVENTS,
            x="Event",
            y="Economic_Loss_Billion_USD",
            color="Affected_People_Millions",
            text_auto=True,
            title="Economic Damages ($ Billions) & Affected Population (Millions)",
            labels={"Economic_Loss_Billion_USD": "Economic Loss ($B)", "Affected_People_Millions": "Affected (M)"},
            color_continuous_scale="Reds"
        )
        fig_events.update_layout(height=400)
        st.plotly_chart(fig_events, use_container_width=True)

    with col_h2:
        st.markdown("#### Provincial Impact Breakdown (2022 Super Floods)")
        fig_prov = px.pie(
            PROVINCIAL_DAMAGE_2022,
            names="Province",
            values="Affected_Millions",
            title="Proportion of Affected Population by Province (2022)",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_prov.update_layout(height=400)
        st.plotly_chart(fig_prov, use_container_width=True)

    st.markdown("#### Provincial Damage Metrics Table (2022 Monsoon)")
    st.dataframe(PROVINCIAL_DAMAGE_2022, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: NATIONWIDE EMERGENCY DIRECTORY
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("📞 Comprehensive Nationwide Pakistan Emergency Directory")
    st.markdown("Filter and search verified helpline contacts across national, provincial, and regional authorities.")

    directory_data = [
        {"Scope": "National", "Province": "All", "Organization": "Rescue 1122", "Role": "Medical, Fire & Disaster Evacuation", "Helpline": "1122"},
        {"Scope": "National", "Province": "All", "Organization": "NDMA", "Role": "National Disaster Management Center", "Helpline": "051-111-157-157"},
        {"Scope": "National", "Province": "All", "Organization": "Edhi Foundation", "Role": "Emergency Ambulance & Relief", "Helpline": "115"},
        {"Scope": "National", "Province": "All", "Organization": "Chhipa Welfare", "Role": "Rescue & Ambulance Operations", "Helpline": "1020"},
        {"Scope": "National", "Province": "All", "Organization": "AlKhidmat Foundation", "Role": "Clean Water & Disaster Relief Camps", "Helpline": "1023"},
        {"Scope": "National", "Province": "All", "Organization": "Red Crescent (PRCS)", "Role": "First Aid & Emergency Shelter", "Helpline": "1030"},
        {"Scope": "National", "Province": "All", "Organization": "Flood Forecasting Division (FFD)", "Role": "Live River & Telemetry Advisories", "Helpline": "042-99200139"},
        {"Scope": "Provincial", "Province": "Punjab", "Organization": "PDMA Punjab", "Role": "Punjab Disaster Control Room", "Helpline": "1129"},
        {"Scope": "Provincial", "Province": "Sindh", "Organization": "PDMA Sindh", "Role": "Sindh Emergency Operations Center", "Helpline": "021-99332005"},
        {"Scope": "Provincial", "Province": "KPK", "Organization": "PDMA KPK", "Role": "KPK Disaster Control Room", "Helpline": "1700 / 0316-4261700"},
        {"Scope": "Provincial", "Province": "Balochistan", "Organization": "PDMA Balochistan", "Role": "Balochistan Disaster Management", "Helpline": "081-9241133"},
        {"Scope": "Provincial", "Province": "Gilgit-Baltistan", "Organization": "GBDMA", "Role": "GB Emergency & GLOF Response", "Helpline": "05811-920830"},
        {"Scope": "Provincial", "Province": "AJK", "Organization": "SDMA AJK", "Role": "State Disaster Management Authority", "Helpline": "05822-921536"},
        {"Scope": "Provincial", "Province": "ICT", "Organization": "Islamabad Admin", "Role": "DC Control Room Islamabad", "Helpline": "051-9108108"}
    ]

    df_dir = pd.DataFrame(directory_data)
    
    # Filter Controls
    selected_prov_filter = st.selectbox("Filter Directory by Province / Territory:", ["All"] + province_list)
    
    if selected_prov_filter != "All":
        df_filtered = df_dir[(df_dir["Province"] == selected_prov_filter) | (df_dir["Province"] == "All")]
    else:
        df_filtered = df_dir

    st.table(df_filtered)

# ---------------------------------------------------------------------------
# TAB 4: ECOSYSTEM BALANCE & MITIGATION
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("🌿 Long-Term Ecological Protection & Risk Mitigation")
    st.markdown("""
    Sustainable flood risk reduction in Pakistan requires combining real-time alerts with proactive ecological restoration:

    #### 1. Upper Catchment & Mountain Slope Protection (GB, KPK, AJK)
    * **Bio-Engineering:** Reforest mountain watersheds with deep-rooted native tree species (Willow, Poplar, Deodar) to anchor topsoil and suppress landslide risks.
    * **Community GLOF Monitoring (CBFEWS):** Establish community early warning watchtowers and automatic acoustic sensors along glacial lakes (e.g., Hassanabad, Passu, Shishper).

    #### 2. Urban Drainage & Sponge City Adaptation (Rawalpindi, Lahore, Karachi, Peshawar)
    * **Sponge City Infrastructure:** Replace non-permeable pavements with porous concrete and urban retention reservoirs to absorb sudden surface runoff.
    * **Encroachment Clearance:** Clear and maintain natural drainage channels (e.g., Nullah Lai, Lyari River, Malir Corridor, Aik Nullah) to preserve maximum flood discharge capacity.

    #### 3. Plain Basins & Coastal Buffer Zones (Sindh, Punjab, Balochistan)
    * **Mangrove Restoration:** Protect and plant coastal mangroves along the Sindh and Makran coasts to dampen storm surges and seawater intrusion.
    * **Wetland Storage Reservoirs:** Preserve and restore natural riverine wetlands along the Indus, Chenab, and Kabul rivers to store peak water discharges naturally during monsoon peaks.
    """)

# ---------------------------------------------------------------------------
# TAB 5: HOW TO USE APP GUIDE
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("📖 User Guide: How to Operate FloodReady AI")
    st.markdown("""
    Welcome to **FloodReady AI**. This application is designed to help Pakistani households, community leaders, and emergency workers prepare for and respond to flood threats effectively.

    ---

    ### 🛠️ Step-by-Step Instructions

    #### 1. Select Your Location
    * Look at the **sidebar on the left**.
    * Select your **Province/Territory** (e.g., *Punjab, Sindh, KPK, GB, etc.*).
    * Choose your specific **District** and **Tehsil / Ravine / Zone**.
    * Enter your exact **Street Address or Landmark** for localized guidance.

    #### 2. Enter Your Household Demographics
    * Specify total household capacity (supports up to **100 members** for joint families or community shelters).
    * Set counts for **Elderly members**, **Children/Infants**, and **Special Needs/Disabled members**.
    * Indicate whether your family has access to a **Motor Vehicle** to get custom parking/evacuation recommendations.

    #### 3. Generate Your Personalized 24-Hour Plan (Tab 1)
    * Click the **"⚡ Generate Personalized 24-Hour Action Plan"** button in Tab 1.
    * The AI engine will analyze your location and demographics to create a timed plan split into:
      * **Phase 1: Pre-Flood Preparation (T-24 Hours)**
      * **Phase 2: Mandatory Evacuation Protocol**
    * Each step is provided in **English** and **Roman Urdu** for easy reading.

    #### 4. Monitor Live Weather & Historical Risk (Tab 2)
    * View real-time 7-day rainfall forecasts and peak probability charts pulled directly from Open-Meteo telemetry for your district.
    * Review historical flood damage trends (2010–2024) to understand regional risk factors.

    #### 5. Access Emergency Helplines (Tab 3)
    * Filter contacts by province to quickly find **Rescue 1122**, **PDMA**, **NDMA**, and non-profit ambulance networks (**Edhi**, **Chhipa**, **AlKhidmat**).

    ---
    *Emergency Tip: Bookmark this application on your mobile browser before severe weather hits.*
    """)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("FloodReady AI Pakistan • Anticipatory Emergency Action Engine • Built with Streamlit, Plotly & Groq AI") 
