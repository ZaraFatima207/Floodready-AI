import os
import requests
import pandas as pd
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
    page_title="FloodReady AI - Pakistan Disaster Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 800; margin-bottom: 0px; }
    .sub-header { font-size: 1.0rem; color: #4B5563; margin-bottom: 20px; }
    .alert-card { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    .info-card { background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    .stButton>button { width: 100%; background-color: #1E3A8A; color: white; font-weight: bold; border-radius: 8px; height: 3em; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. SYSTEM PROMPT DEFINITION
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are FloodReady AI, a hyper-personalized disaster response and preparedness engine operating in Pakistan. Your task is to transform raw disaster management advisories and hydrometeorological data into structured, timed, and actionable 24-hour preparedness and evacuation plans tailored to specific household profiles and location-based hazards.

### INPUT CONTEXT
- USER LOCATION: {location}
- HOUSEHOLD PROFILE: {household_profile}
- RETRIEVED ADVISORIES & EMERGENCY DATA:
{context}

### OPERATIONAL DIRECTIVES & GUARDRAILS
1. STRICT ADHERENCE TO RETRIEVED DATA:
   - Rely strictly on the provided context for official emergency helplines, shelter locations, and evacuation zones.
   - Do NOT invent shelter addresses or phone numbers. If local shelter information is missing from the context, direct the user to contact Rescue 1122 or the District Administration.
2. DEMOGRAPHIC PERSONALIZATION:
   - Vulnerable Members (Elderly / Children / Infants): Explicitly sequence medical supplies, mobility assistance, formula, and early evacuation priorities before general task lists.
   - Vehicle Protocols: Provide concrete actions for parking or securing vehicles on elevated ground away from local drainage corridors (e.g., Nullah Lai, river beds).
3. DUAL-LANGUAGE DELIVERY:
   - Present every action item in clear English, followed immediately by conversational Roman Urdu (Urdu in Latin script) to ensure accessibility across diverse literacy levels.
4. REGIONAL HAZARD ADAPTATION:
   - High-Altitude / Mountainous (GB & KPK): Emphasize GLOF acoustic warnings, high-ground foot migration, and rockslide safety.
   - Urban & Plain Basins (Rawalpindi / Punjab / Sindh): Emphasize main breaker electrical disconnection, drain spillover containment, waterborne disease safety, and barrage discharge timelines.

### OUTPUT FORMAT REQUIREMENTS
Format your output cleanly using Markdown with the following structure:
1. ⚠️ LIVE THREAT & HAZARD ASSESSMENT
2. 📋 PERSONALIZED 24-HOUR PREPAREDNESS PLAN
   - Phase 1: Pre-Flood Preparation (T-24 Hours) [English & Roman Urdu]
   - Phase 2: Evacuation Protocol (On Advisory) [English & Roman Urdu]
3. 📞 OFFICIAL HELPLINES & SHELTER DIRECTORY
"""

# ---------------------------------------------------------------------------
# 3. GEOGRAPHIC COORDINATES & LIVE WEATHER FETCHING
# ---------------------------------------------------------------------------
DISTRICT_COORDINATES = {
    "Rawalpindi": {"lat": 33.5970, "lon": 73.0439},
    "Swat": {"lat": 34.7717, "lon": 72.3602},
    "Hunza": {"lat": 36.3167, "lon": 74.6500},
    "D.G. Khan": {"lat": 30.0561, "lon": 70.6348},
    "Sukkur": {"lat": 27.7052, "lon": 68.8574},
    "Other (National)": {"lat": 33.6844, "lon": 73.0479}
}

@st.cache_data(ttl=1800)
def fetch_weather_forecast(lat: float, lon: float):
    """Fetches 7-day daily and 24-hour hourly weather forecast from Open-Meteo."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weathercode&hourly=temperature_2m,precipitation_probability,rain&timezone=Asia%2FKarachi"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.warning(f"Could not fetch live weather data: {e}")
    return None

def interpret_weather_code(code: int) -> str:
    """Maps WMO Weather Codes to human-readable strings."""
    codes = {
        0: "Clear Sky ☀️",
        1: "Mainly Clear 🌤️", 2: "Partly Cloudy ⛅", 3: "Overcast ☁️",
        45: "Foggy 🌫️", 51: "Light Drizzle 🌧️", 61: "Slight Rain 🌧️",
        63: "Moderate Rain 🌧️", 65: "Heavy Rain ⛈️", 80: "Rain Showers 🌦️",
        95: "Thunderstorm 🌩️", 96: "Thunderstorm with Hail ⛈️"
    }
    return codes.get(code, "Cloudy / Variable ⛅")

# ---------------------------------------------------------------------------
# 4. KNOWLEDGE BASE DATASET (PAKISTAN ADVISORIES & DIRECTORY)
# ---------------------------------------------------------------------------
PAKISTAN_DISASTER_KNOWLEDGE_BASE = [
    {
        "district": "Rawalpindi",
        "authority": "PDMA Punjab / District Admin Rawalpindi",
        "content": """Rawalpindi Nullah Lai Warning System: Monitoring points at Cattledam and Gawalmandi.
Alert Level 1 (11 ft): Monitoring; Alert Level 2 (14 ft): Standby Evacuation; Critical Level (18 ft): Mandatory Evacuation.
Safe Relief Shelters: Govt Gordon College Rawalpindi, Govt High School Westridge, Govt College Asghar Mall.
Helplines: Rescue 1122, Rawalpindi District Control Room: 051-929296, PDMA Punjab Helpline: 1129."""
    },
    {
        "district": "Swat",
        "authority": "PDMA KPK",
        "content": """Swat River Basin & Flash Flood Zones: Bahrain, Kalam, Mingora riverbanks, Khwazakhela.
Safe Relief Shelters: Govt High School Kalam, Govt Degree College Mingora.
Helplines: Rescue 1122, PDMA KPK Emergency Hotline: 1700, WhatsApp: 0316-4261700.
Evacuation Strategy: Move away from riverbanks to elevated ridges immediately upon heavy rain upstream."""
    },
    {
        "district": "Hunza",
        "authority": "GBDMA / AKAH",
        "content": """GLOF Hazards & Landslide Corridors: Passu, Shishper Glacier Lake, Hassanabad Bridge area, Bagrot Valley, Ghizer.
Community Early Warning: Local siren system + Acoustic sensors monitored by Aga Khan Agency for Habitat (AKAH).
Relief Shelters: AKAH Community Centers, Govt Boys High School Hunza.
Helplines: GBDMA Control Room: 05811-920830, Rescue 1122 GB."""
    },
    {
        "district": "D.G. Khan",
        "authority": "NDMA / PDMA Punjab",
        "content": """Hill Torrent & Riverine Floods: D.G. Khan, Rajanpur (Kaha Sultan & Mithawan torrents), Muzaffargarh.
Safe Shelters: District Complex D.G. Khan, Govt High Schools in elevated union councils.
Helplines: Rescue 1122, PDMA Punjab: 1129, NDMA Helpline: 051-111-157-157."""
    },
    {
        "district": "Sukkur",
        "authority": "PDMA Sindh",
        "content": """Indus River Super Flood Corridors: Sukkur Barrage, Larkana, Dadu, Badin.
Relief Shelters: Govt Degree Colleges and Tents managed by PDMA Sindh.
Helplines: PDMA Sindh Control Room: 021-99332005, Rescue 1122 Sindh, Edhi Emergency: 115."""
    },
    {
        "district": "National",
        "authority": "National Directory",
        "content": """National Official Emergency Contacts:
- NDMA National Emergency Helpline: 051-111-157-157
- Rescue 1122 (Search, Medical, & Fire Evacuation): 1122
- Edhi Ambulance & Emergency Relief: 115
- AlKhidmat Foundation Field Relief & Water: 1023
- Pakistan Red Crescent Society (PRCS): 1030
- Flood Forecasting Division (FFD) Hotline: 042-99200139"""
    }
]

# ---------------------------------------------------------------------------
# 5. VECTOR STORE LOADER
# ---------------------------------------------------------------------------
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = [
        Document(
            page_content=item["content"],
            metadata={"district": item["district"], "authority": item["authority"]}
        )
        for item in PAKISTAN_DISASTER_KNOWLEDGE_BASE
    ]
    return FAISS.from_documents(documents, embeddings)

# ---------------------------------------------------------------------------
# 6. SIDEBAR: USER INPUTS & GROQ CONFIGURATION
# ---------------------------------------------------------------------------
st.sidebar.title("🌊 FloodReady AI")
st.sidebar.markdown("**Personalized Disaster Preparedness Engine**")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=st.secrets.get("GROQ_API_KEY", ""),
    help="Get a free key from console.groq.com"
)

# Active Fast Non-Llama Production Models on Groq
FAST_NON_LLAMA_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini"
]

selected_model = st.sidebar.selectbox(
    "LLM Model (Fast Non-Llama Endpoints)",
    FAST_NON_LLAMA_MODELS,
    index=0,
    help="Selected active non-Llama model hosted on Groq."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Household & Location Context")

district = st.sidebar.selectbox(
    "District / Region",
    ["Rawalpindi", "Swat", "Hunza", "D.G. Khan", "Sukkur", "Other (National)"]
)

proximity_tag = st.sidebar.text_input(
    "Specific Neighborhood / Stream",
    value="Near Nullah Lai (Arya Mohallah)",
    help="e.g., Near Nullah Lai, Swat River Bank, Passu Village"
)

total_members = st.sidebar.number_input("Total Household Members", min_value=1, max_value=20, value=5)
elderly_count = st.sidebar.number_input("Elderly Members (60+ yrs)", min_value=0, max_value=10, value=1)
children_count = st.sidebar.number_input("Children / Infants", min_value=0, max_value=10, value=2)
has_vehicle = st.sidebar.checkbox("Household Has Motor Vehicle", value=True)

# ---------------------------------------------------------------------------
# 7. MAIN INTERFACE & TABS
# ---------------------------------------------------------------------------
st.markdown("<div class='main-header'>🌊 FloodReady AI Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Anticipatory Disaster Preparedness & Live Weather Intelligence for Pakistan</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 24-Hour Action Plan", 
    "🌤️ Weather & Flood Analytics", 
    "📞 Emergency Directory", 
    "🌿 Ecosystem Protection"
])

# ---------------------------------------------------------------------------
# TAB 1: 24-HOUR ACTION PLAN
# ---------------------------------------------------------------------------
with tab1:
    st.markdown(f"""
    <div class='alert-card'>
        <strong>⚠️ Selected Zone:</strong> {district} ({proximity_tag})<br/>
        <strong>👥 Family Dynamics:</strong> {total_members} Members ({elderly_count} Elderly, {children_count} Children) | <strong>Vehicle:</strong> {'Yes' if has_vehicle else 'No'}
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Generate 24-Hour Personalized Plan"):
        if not groq_api_key:
            st.error("Please enter a valid Groq API Key in the sidebar or set `GROQ_API_KEY` in Streamlit Secrets.")
        else:
            with st.spinner("Retrieving advisories & generating personalized plan..."):
                try:
                    # 1. Retrieve Context from FAISS Vector Store
                    vector_store = load_vector_store()
                    query = f"Flood safety evacuation shelter emergency helpline {district} {proximity_tag}"
                    retrieved_docs = vector_store.similarity_search(query, k=3)
                    
                    context_str = "\n\n".join([
                        f"[Source: {doc.metadata['authority']} - {doc.metadata['district']}]\n{doc.page_content}"
                        for doc in retrieved_docs
                    ])
                    
                    household_summary = (
                        f"Total Members: {total_members} | "
                        f"Elderly: {elderly_count} | "
                        f"Children: {children_count} | "
                        f"Vehicle Access: {'Yes' if has_vehicle else 'No'}"
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", SYSTEM_PROMPT),
                        ("human", "Generate my personalized preparedness plan.")
                    ])

                    # 2. Execution Loop with Multi-Model Non-Llama Fallback
                    models_to_try = [selected_model] + [m for m in FAST_NON_LLAMA_MODELS if m != selected_model]
                    response_text = None
                    last_error = None
                    successful_model = None

                    for model_name in models_to_try:
                        try:
                            llm = ChatGroq(
                                groq_api_key=groq_api_key,
                                model_name=model_name,
                                temperature=0.1
                            )
                            chain = prompt | llm | StrOutputParser()
                            response_text = chain.invoke({
                                "location": f"{district} - {proximity_tag}",
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
                            st.info(f"Note: Selected model was unavailable. Plan generated using fallback model: `{successful_model}`")
                        st.markdown("---")
                        st.markdown(response_text)
                    else:
                        st.error(f"Error executing plan generation across models: {last_error}")
                    
                except Exception as e:
                    st.error(f"System Error: {str(e)}")

# ---------------------------------------------------------------------------
# TAB 2: WEATHER & FLOOD RISK GRAPHICAL ANALYTICS
# ---------------------------------------------------------------------------
with tab2:
    st.subheader(f"🌦️ Live Weather & Flood Prediction for {district}")
    
    coords = DISTRICT_COORDINATES.get(district, DISTRICT_COORDINATES["Other (National)"])
    weather_data = fetch_weather_forecast(coords["lat"], coords["lon"])
    
    if weather_data and "daily" in weather_data:
        daily = weather_data["daily"]
        dates = daily["time"]
        max_temps = daily["temperature_2m_max"]
        min_temps = daily["temperature_2m_min"]
        precip_sum = daily["precipitation_sum"]
        precip_prob = daily["precipitation_probability_max"]
        weather_codes = daily["weathercode"]
        
        # Display Metric Widgets
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Today's Weather", interpret_weather_code(weather_codes[0]))
        col2.metric("Today's Max Temp", f"{max_temps[0]} °C")
        col3.metric("Total 7-Day Expected Rain", f"{sum(precip_sum):.1f} mm")
        col4.metric("Peak Rain Probability", f"{max(precip_prob)} %")
        
        st.markdown("---")
        st.markdown("### 📊 7-Day Weather Forecast & Rainfall Probability")
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.12,
            subplot_titles=("7-Day Temperature Range (°C)", "Daily Expected Precipitation (mm) & Rain Probability (%)")
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
            go.Bar(x=dates, y=precip_sum, name="Rainfall (mm)", marker_color="#1D4ED8", opacity=0.7),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=precip_prob, name="Rain Prob (%)", line=dict(color="#10B981", width=2), mode="lines+markers"),
            row=2, col=1
        )
        
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ⏱️ Next 24-Hour Hourly Precipitation Risk Timeline")
        if "hourly" in weather_data:
            hourly = weather_data["hourly"]
            df_hourly = pd.DataFrame({
                "Time": [t.replace("T", " ") for t in hourly["time"][:24]],
                "Temperature (°C)": hourly["temperature_2m"][:24],
                "Rain Probability (%)": hourly["precipitation_probability"][:24],
                "Rain Volume (mm)": hourly["rain"][:24]
            })
            
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=df_hourly["Time"], 
                y=df_hourly["Rain Volume (mm)"], 
                name="Rain Volume (mm)", 
                marker_color="#2563EB"
            ))
            fig_hourly.add_trace(go.Scatter(
                x=df_hourly["Time"], 
                y=df_hourly["Rain Probability (%)"], 
                name="Rain Prob (%)", 
                line=dict(color="#D97706", width=2)
            ))
            fig_hourly.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Time", hovermode="x unified")
            st.plotly_chart(fig_hourly, use_container_width=True)
            
    else:
        st.info("Select a district from the sidebar to view weather predictions.")

# ---------------------------------------------------------------------------
# TAB 3: DIRECTORY
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("📞 Verified Emergency Directory & Disaster Agencies")
    directory_data = [
        {"Organization": "Rescue 1122", "Role": "Primary Medical, Fire & Evacuation Rescue", "Helpline": "1122", "Coverage": "Punjab, KPK, GB, AJK"},
        {"Organization": "NDMA", "Role": "National Disaster Management Authority", "Helpline": "051-111-157-157", "Coverage": "National"},
        {"Organization": "PDMA KPK", "Role": "KPK Provincial Response & Emergency Control", "Helpline": "1700 / 0316-4261700", "Coverage": "Khyber Pakhtunkhwa"},
        {"Organization": "PDMA Punjab", "Role": "Punjab Disaster Control Room", "Helpline": "1129", "Coverage": "Punjab"},
        {"Organization": "PDMA Sindh", "Role": "Sindh Provincial Disaster Control Room", "Helpline": "021-99332005", "Coverage": "Sindh"},
        {"Organization": "GBDMA", "Role": "Gilgit-Baltistan Disaster Management Authority", "Helpline": "05811-920830", "Coverage": "Gilgit-Baltistan"},
        {"Organization": "Edhi Foundation", "Role": "Ambulance, Medical & Emergency Relief", "Helpline": "115", "Coverage": "National"},
        {"Organization": "AlKhidmat Foundation", "Role": "Field Camps, Cooked Meals & Clean Water", "Helpline": "1023", "Coverage": "National"},
        {"Organization": "Pakistan Red Crescent (PRCS)", "Role": "First-Aid & Emergency Shelter Kits", "Helpline": "1030", "Coverage": "National"},
        {"Organization": "Flood Forecasting Division (FFD)", "Role": "Live River & Rain Telemetry", "Helpline": "042-99200139", "Coverage": "National"}
    ]
    st.table(directory_data)

# ---------------------------------------------------------------------------
# TAB 4: ECOSYSTEM BALANCE
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("🌿 Ecosystem Balance & Long-Term Risk Mitigation")
    st.markdown("""
    Anticipatory flood safety requires long-term ecological restoration and community-level preventative actions alongside short-term alerts:

    #### 1. Mountain Slope Stabilization (Gilgit-Baltistan & Upper KPK)
    * **Bio-Engineering:** Replant native deep-root tree species (e.g., Willow, Poplar) along vulnerable slopes to bind soil and suppress landslide risks.
    * **GLOF Early Warning:** Install acoustic and vibration telemetry along glacial stream beds paired with community watchtowers (CBFEWS).

    #### 2. Urban & Plain Basin Drainage (Rawalpindi, Lahore, Karachi)
    * **Sponge City Infrastructure:** Replace impermeable concrete surfaces with permeable pavements and urban retention basins to absorb sudden surface runoff.
    * **Riverbank Buffer Enforcement:** Restrict illegal encroachment along natural floodplains (e.g., Nullah Lai, Swat Riverbed).

    #### 3. Coastal & Riverine Ecosystems (Sindh & South Punjab)
    * **Mangrove Ecosystem Protection:** Expand coastal mangrove plantations in Sindh to attenuate tidal surges and shoreline erosion.
    * **Wetland Restoration:** Preserve natural riverine wetlands to serve as secondary storage reservoirs during peak barrage discharges.
    """)
