import os
import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodReady AI - Pakistan Disaster Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Emergency UI
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
# 3. KNOWLEDGE BASE DATASET (PAKISTAN ADVISORIES & DIRECTORY)
# ---------------------------------------------------------------------------
PAKISTAN_DISASTER_KNOWLEDGE_BASE = [
    # Rawalpindi & Nullah Lai Basin
    {
        "district": "Rawalpindi",
        "authority": "PDMA Punjab / District Admin Rawalpindi",
        "content": """Rawalpindi Nullah Lai Warning System: Monitoring points at Cattledam and Gawalmandi.
Alert Level 1 (11 ft): Monitoring; Alert Level 2 (14 ft): Standby Evacuation; Critical Level (18 ft): Mandatory Evacuation.
Safe Relief Shelters: Govt Gordon College Rawalpindi, Govt High School Westridge, Govt College Asghar Mall.
Helplines: Rescue 1122, Rawalpindi District Control Room: 051-929296, PDMA Punjab Helpline: 1129."""
    },
    # Swat & Upper KPK
    {
        "district": "Swat",
        "authority": "PDMA KPK",
        "content": """Swat River Basin & Flash Flood Zones: Bahrain, Kalam, Mingora riverbanks, Khwazakhela.
Safe Relief Shelters: Govt High School Kalam, Govt Degree College Mingora.
Helplines: Rescue 1122, PDMA KPK Emergency Hotline: 1700, WhatsApp: 0316-4261700.
Evacuation Strategy: Move away from riverbanks to elevated ridges immediately upon heavy rain upstream."""
    },
    # Gilgit-Baltistan (GLOF & Landslides)
    {
        "district": "Hunza",
        "authority": "GBDMA / AKAH",
        "content": """GLOF Hazards & Landslide Corridors: Passu, Shishper Glacier Lake, Hassanabad Bridge area, Bagrot Valley, Ghizer.
Community Early Warning: Local siren system + Acoustic sensors monitored by Aga Khan Agency for Habitat (AKAH).
Relief Shelters: AKAH Community Centers, Govt Boys High School Hunza.
Helplines: GBDMA Control Room: 05811-920830, Rescue 1122 GB."""
    },
    # Plain Basins (South Punjab & Sindh)
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
    # National Helplines & Field Relief Directory
    {
        "district": "National",
        "authority": "National Directory",
        "content": """National Official Emergency Contacts:
- NDMA National Emergency Helpline: 051-111-157-157
- Rescue 1122 (Search, Medical, & Fire Evacuation): 1122
- Edhi Ambulance & Emergency Relief: 115
- AlKhidmat Foundation Field Relief & Water: 1023
- Pakistan Red Crescent Society (PRCS): 1030
- Flood Forecasting Division (FFD) Hotline: 042-99200139
Ecosystem & Long-Term Measures: Replanting Willow & Poplar trees on slopes (GB/KPK), protecting mangrove buffers in Sindh, implementing porous pavements in urban centers, and preserving river floodplains."""
    }
]

# ---------------------------------------------------------------------------
# 4. INITIALIZE VECTOR STORE & CACHE RESOURCE
# ---------------------------------------------------------------------------
@st.cache_resource
def load_vector_store():
    """Builds an in-memory FAISS vector index using CPU-friendly HuggingFace Embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    documents = [
        Document(
            page_content=item["content"],
            metadata={"district": item["district"], "authority": item["authority"]}
        )
        for item in PAKISTAN_DISASTER_KNOWLEDGE_BASE
    ]
    
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# ---------------------------------------------------------------------------
# 5. SIDEBAR: USER INPUTS & GROQ CONFIGURATION
# ---------------------------------------------------------------------------
st.sidebar.title("🌊 FloodReady AI")
st.sidebar.markdown("**Personalized Disaster Preparedness Engine**")

# Groq API Key Handling (Reads from Streamlit Secrets or manual input)
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=st.secrets.get("GROQ_API_KEY", ""),
    help="Get a free key from console.groq.com"
)

selected_model = st.sidebar.selectbox(
    "LLM Architecture (Groq Free Tier)",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    help="llama-3.3-70b-versatile delivers high reasoning quality; 8b-instant provides ultra-fast response speed."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Household & Location Context")

district = st.sidebar.selectbox(
    "District / Region",
    ["Rawalpindi", "Swat", "Hunza", "D.G. Khan", "Sukkur", "Other (National)"]
)

proximity_tag = st.sidebar.text_input(
    "Specific Neighborhood / Drainage Stream",
    value="Near Nullah Lai (Arya Mohallah)",
    help="e.g., Near Nullah Lai, Swat River Bank, Passu Village, Hill Torrent Basin"
)

total_members = st.sidebar.number_input("Total Household Members", min_value=1, max_value=20, value=5)
elderly_count = st.sidebar.number_input("Elderly Members (60+ yrs)", min_value=0, max_value=10, value=1)
children_count = st.sidebar.number_input("Children / Infants", min_value=0, max_value=10, value=2)
has_vehicle = st.sidebar.checkbox("Household Has Motor Vehicle", value=True)

# ---------------------------------------------------------------------------
# 6. MAIN INTERFACE & TABS
# ---------------------------------------------------------------------------
st.markdown("<div class='main-header'>🌊 FloodReady AI Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Anticipatory Disaster Preparedness & Action Plan for Pakistan</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 24-Hour Action Plan", "📞 Emergency Directory", "🌿 Ecosystem Protection"])

with tab1:
    # Live Hazard Banner
    st.markdown(f"""
    <div class='alert-card'>
        <strong>⚠️ Selected Zone:</strong> {district} ({proximity_tag})<br/>
        <strong>👥 Family Dynamics:</strong> {total_members} Members ({elderly_count} Elderly, {children_count} Children) | <strong>Vehicle:</strong> {'Yes' if has_vehicle else 'No'}
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Generate 24-Hour Personalized Plan"):
        if not groq_api_key:
            st.error("Please enter a valid Groq API Key in the sidebar or set it in Streamlit Secrets (`GROQ_API_KEY`).")
        else:
            with st.spinner("Retrieving official advisories & generating dual-language action plan..."):
                try:
                    # 1. Retrieve Context from FAISS
                    vector_store = load_vector_store()
                    query = f"Flood safety evacuation shelter emergency helpline {district} {proximity_tag}"
                    retrieved_docs = vector_store.similarity_search(query, k=3)
                    
                    context_str = "\n\n".join([
                        f"[Source: {doc.metadata['authority']} - {doc.metadata['district']}]\n{doc.page_content}"
                        for doc in retrieved_docs
                    ])
                    
                    # 2. Setup Groq Chat Model
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name=selected_model,
                        temperature=0.1
                    )
                    
                    # 3. Format Prompt & Execute Chain
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", SYSTEM_PROMPT),
                        ("human", "Generate my personalized preparedness plan.")
                    ])
                    
                    chain = prompt | llm | StrOutputParser()
                    
                    household_summary = (
                        f"Total Members: {total_members} | "
                        f"Elderly: {elderly_count} | "
                        f"Children: {children_count} | "
                        f"Vehicle Access: {'Yes' if has_vehicle else 'No'}"
                    )
                    
                    response = chain.invoke({
                        "location": f"{district} - {proximity_tag}",
                        "household_profile": household_summary,
                        "context": context_str
                    })
                    
                    # Display Result
                    st.markdown("---")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Error generating plan: {str(e)}")

with tab2:
    st.subheader("📞 Verified Emergency Directory & Disaster Agencies")
    st.write("Access direct helplines for rescue operations, field relief, and flood updates across Pakistan.")
    
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

with tab3:
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
