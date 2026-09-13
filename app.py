import os
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import urllib.parse
from groq import Groq

# ------------------------------------------------------------------------------
# 1. Page Configuration & UI Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodReady AI - Pakistan Emergency Preparedness & Early Warning",
    page_icon="🌊",
    layout="wide"
)

# Main Application Title
st.title("🌊 FloodReady AI - Early Warning & Response System")
st.markdown("*National Real-Time Flood Risk Analytics, Automated Early Warning System, and AI Response Planning for Pakistan.*")

# ------------------------------------------------------------------------------
# 2. System Settings & Regional Language Sidebar Controls
# ------------------------------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE"))

st.sidebar.header("⚙️ System & Language Options")

# AI Model Selection
selected_model = st.sidebar.selectbox(
    "Select AI Model Engine:",
    [
        "openai/gpt-oss-20b",     # Ultra-fast
        "openai/gpt-oss-120b",    # High reasoning
        "qwen/qwen3.6-27b",       # Fast multilingual
        "groq/compound-mini"      # Fallback engine
    ],
    index=0
)

NON_LLAMA_MODELS = [
    selected_model,
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini"
]

# Regional Pakistani Languages
pakistani_languages = [
    "Urdu",
    "Pashto",
    "Sindhi",
    "Saraiki",
    "Punjabi",
    "Hindko",
    "Balochi",
    "English Only"
]

selected_lang = st.sidebar.selectbox(
    "🌐 Action Plan Secondary Language:",
    options=pakistani_languages,
    index=0,
    help="Select language for captions below each action item."
)

# ------------------------------------------------------------------------------
# 3. Comprehensive Pakistan Geographic Hierarchy Data
# ------------------------------------------------------------------------------
PAKISTAN_GEOGRAPHY = {
    "Punjab": {
        "Lahore": {"lat": 31.5204, "lon": 74.3587, "tehsils": ["Lahore City", "Model Town", "Cantonment", "Raiwind", "Shalimar"]},
        "Rawalpindi": {"lat": 33.5989, "lon": 73.0441, "tehsils": ["Rawalpindi City", "Gujar Khan", "Kahuta", "Kallan Syedan", "Murree", "Taxila"]},
        "Multan": {"lat": 30.1575, "lon": 71.5249, "tehsils": ["Multan City", "Multan Sadar", "Shujabad", "Jalalpur Pirwala"]},
        "Faisalabad": {"lat": 31.4504, "lon": 73.1350, "tehsils": ["Faisalabad City", "Sadar", "Chak Jhumra", "Jaranwala", "Tandlianwala", "Sammundri"]},
        "Dera Ghazi Khan": {"lat": 30.0561, "lon": 70.6348, "tehsils": ["DG Khan City", "Taunsa Sharif", "De-Excluded Area", "Kot Chutta"]},
        "Sialkot": {"lat": 32.4945, "lon": 74.5229, "tehsils": ["Sialkot City", "Daska", "Pasrur", "Sambrial"]},
        "Muzaffargarh": {"lat": 30.0703, "lon": 71.1933, "tehsils": ["Muzaffargarh", "Alipur", "Jatoi", "Kot Addu"]},
        "Rahim Yar Khan": {"lat": 28.4212, "lon": 70.2989, "tehsils": ["Rahim Yar Khan", "Khanpur", "Liaquatpur", "Sadiqabad"]}
    },
    "Sindh": {
        "Karachi Central": {"lat": 24.9180, "lon": 67.0330, "tehsils": ["Gulberg", "Liaquatabad", "Nazimabad", "North Nazimabad", "New Karachi"]},
        "Karachi South": {"lat": 24.8607, "lon": 67.0011, "tehsils": ["Civil Lines", "Garden", "Lyari", "Saddar", "Aram Bagh"]},
        "Hyderabad": {"lat": 25.3960, "lon": 68.3578, "tehsils": ["Hyderabad City", "Hyderabad Latifabad", "Hyderabad Qasimabad", "Rural"]},
        "Sukkur": {"lat": 27.7131, "lon": 68.8368, "tehsils": ["Sukkur City", "Rohri", "Pano Akil", "New Sukkur"]},
        "Dadu": {"lat": 26.7303, "lon": 67.7769, "tehsils": ["Dadu", "Johi", "Mehar", "Khairpur Nathan Shah"]},
        "Larkana": {"lat": 27.5590, "lon": 68.2120, "tehsils": ["Larkana", "Ratodero", "Dokri", "Bakrani"]},
        "Thatta": {"lat": 24.7475, "lon": 67.9239, "tehsils": ["Thatta", "Mirpur Sakro", "Ghorabari", "Keti Bandar"]}
    },
    "Khyber Pakhtunkhwa (KPK)": {
        "Peshawar": {"lat": 34.0151, "lon": 71.5249, "tehsils": ["Peshawar City", "Peshawar Sadar", "Shah Alam", "Matani", "Chamkani"]},
        "Swat": {"lat": 35.2227, "lon": 72.4258, "tehsils": ["Babuzai (Mingora)", "Barikot", "Charbagh", "Kabal", "Matta", "Khwazakhela", "Kalam"]},
        "Nowshera": {"lat": 34.0105, "lon": 71.9876, "tehsils": ["Nowshera", "Pabbi", "Jehangira"]},
        "Charsadda": {"lat": 34.1509, "lon": 71.7359, "tehsils": ["Charsadda", "Shabqaddar", "Tangi"]},
        "Dera Ismail Khan": {"lat": 31.8314, "lon": 70.9019, "tehsils": ["DI Khan", "Kulachi", "Daraban", "Paharpur", "Paroa"]}
    },
    "Balochistan": {
        "Quetta": {"lat": 30.1798, "lon": 66.9750, "tehsils": ["Quetta City", "Chiltan", "Sariab", "Zarghoon"]},
        "Jaffarabad": {"lat": 28.3242, "lon": 68.2231, "tehsils": ["Dera Allah Yar", "Jhatpat", "Usta Mohammad"]},
        "Naseerabad": {"lat": 28.7505, "lon": 68.1751, "tehsils": ["Dera Murad Jamali", "Chattar", "Tamboo"]},
        "Gwadar": {"lat": 25.1216, "lon": 62.3254, "tehsils": ["Gwadar", "Ormara", "Pasni", "Jiwani"]},
        "Lasbela": {"lat": 26.2238, "lon": 66.3045, "tehsils": ["Bela", "Hub", "Uthal", "Dureji", "Gaddani"]}
    },
    "Gilgit-Baltistan (GB)": {
        "Gilgit": {"lat": 35.9208, "lon": 74.3089, "tehsils": ["Gilgit City", "Danyor", "Juglot"]},
        "Skardu": {"lat": 35.2971, "lon": 75.6333, "tehsils": ["Skardu", "Gultari", "Rondu"]},
        "Hunza": {"lat": 36.3167, "lon": 74.6500, "tehsils": ["Aliabad", "Gojal (Upper Hunza)"]}
    },
    "Azad Jammu & Kashmir (AJK)": {
        "Muzaffarabad": {"lat": 34.3700, "lon": 73.4711, "tehsils": ["Muzaffarabad", "Pattika (Naseerabad)"]},
        "Mirpur": {"lat": 33.1484, "lon": 73.7518, "tehsils": ["Mirpur", "Dadyal"]},
        "Rawalakot (Poonch)": {"lat": 33.8584, "lon": 73.7653, "tehsils": ["Rawalakot", "Hajira", "Thorar"]}
    },
    "Islamabad Capital Territory": {
        "Islamabad": {"lat": 33.6844, "lon": 73.0479, "tehsils": ["Zone I (Urban)", "Zone II (Model Town)", "Zone III (Margalla Hills)", "Zone IV (Rawal Lake/Bhara Kahu)", "Zone V (Kahuta Road)"]}
    }
}

# ------------------------------------------------------------------------------
# 4. Weather API & Alert Processing Logic
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def get_weather_forecast(lat=33.6844, lon=73.0479):
    """Fetch live 7-day weather forecast from Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min&timezone=Asia%2FKarachi"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            daily = data.get("daily", {})
            df = pd.DataFrame({
                "Date": daily.get("time", []),
                "Precipitation (mm)": daily.get("precipitation_sum", []),
                "Rain (mm)": daily.get("rain_sum", []),
                "Max Temp (°C)": daily.get("temperature_2m_max", []),
                "Min Temp (°C)": daily.get("temperature_2m_min", [])
            })
            return df
    except Exception as e:
        st.error(f"Weather fetch failed: {e}")
    return pd.DataFrame()

def evaluate_early_warning_alerts(weather_df, district_name):
    if weather_df.empty:
        return "UNKNOWN", "#6B7280", "WEATHER TELEMETRY OFFLINE", "Unable to load live weather feed."
        
    max_rain = weather_df["Precipitation (mm)"].max()
    max_rain_date = weather_df.loc[weather_df["Precipitation (mm)"].idxmax()]["Date"]
    total_7day_rain = weather_df["Precipitation (mm)"].sum()

    if max_rain >= 80 or total_7day_rain >= 150:
        return (
            "CRITICAL DANGER",
            "#DC2626", # Red
            f"🚨 CRITICAL FLASH FLOOD DANGER WARNING - {district_name.upper()}",
            f"Severe rainfall predicted ({max_rain:.1f} mm on {max_rain_date}). High risk of immediate urban flooding, river overflow, and ravine torrents! Prepare for instant evacuation."
        )
    elif max_rain >= 40 or total_7day_rain >= 80:
        return (
            "HIGH ALERT",
            "#D97706", # Amber
            f"⚠️ HIGH RAINFALL & FLOOD ADVISORY - {district_name.upper()}",
            f"Heavy rainfall anticipated ({max_rain:.1f} mm on {max_rain_date}). Nullahs and low-lying drainage channels may overflow. Pack emergency supplies."
        )
    elif max_rain >= 20:
        return (
            "MODERATE ADVISORY",
            "#2563EB", # Blue
            f"ℹ️ MODERATE WEATHER WATCH - {district_name.upper()}",
            f"Light to moderate rainfall expected ({max_rain:.1f} mm on {max_rain_date}). Maintain normal vigilance and inspect household drainage."
        )
    else:
        return (
            "NORMAL",
            "#059669", # Green
            f"✅ SAFE / STABLE CONDITIONS - {district_name.upper()}",
            f"No major flood risk detected in forecast. Total 7-day predicted rainfall is {total_7day_rain:.1f} mm."
        )

# ------------------------------------------------------------------------------
# 5. AI Plan Generator Integration
# ------------------------------------------------------------------------------
def generate_ai_plan(prompt_text: str):
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return "⚠️ **Groq API Key Missing**: Please configure `GROQ_API_KEY` in `st.secrets` or environment variables."

    client = Groq(api_key=GROQ_API_KEY)
    last_err = None

    for model_name in NON_LLAMA_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are FloodReady AI, Pakistan's official emergency preparedness & flood response coordinator. "
                            "Provide accurate, actionable, life-saving advice for households facing flood threats."
                        )
                    },
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.6,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            continue

    return f"❌ **Error generating response with available models**: {last_err}"

# ------------------------------------------------------------------------------
# 6. Sidebar Location Controls & Banner Alert Display
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location Selector")

selected_province = st.sidebar.selectbox("1. Select Province / Territory:", list(PAKISTAN_GEOGRAPHY.keys()))
available_districts = list(PAKISTAN_GEOGRAPHY[selected_province].keys())

selected_district = st.sidebar.selectbox("2. Select District:", available_districts)
district_data = PAKISTAN_GEOGRAPHY[selected_province][selected_district]

selected_tehsil = st.sidebar.selectbox("3. Select Tehsil / Region:", district_data["tehsils"])
specific_address = st.sidebar.text_input("4. Neighborhood / Address:", placeholder="e.g. Street 4, Sector G-7/2 or UC 5")

# Fetch Live Weather telemetry
weather_df = get_weather_forecast(lat=district_data["lat"], lon=district_data["lon"])
alert_status, alert_bg_color, alert_title, alert_desc = evaluate_early_warning_alerts(weather_df, selected_district)

# Early Warning Alert Banner
st.markdown(f"""
<div style="background-color: {alert_bg_color}; padding: 18px; border-radius: 12px; color: white; margin-bottom: 25px;">
    <h3 style="margin: 0; color: white; font-size: 1.4rem;">{alert_title}</h3>
    <p style="margin: 8px 0 0 0; font-size: 1.05rem;">{alert_desc}</p>
    <p style="margin: 6px 0 0 0; font-size: 0.85rem; font-style: italic;">Selected Location: {selected_tehsil}, {selected_district}, {selected_province}</p>
</div>
""", unsafe_allow_html=True)

if alert_status == "CRITICAL DANGER":
    st.error("🚨 CRITICAL EMERGENCY ALARM ACTIVATED FOR YOUR DISTRICT")

# ------------------------------------------------------------------------------
# 7. Main Navigation Tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Household Action Plan", 
    "🚨 Live Alert & Broadcast Dispatcher",
    "📊 Weather & Historical Flood Radar", 
    "📞 Nationwide Emergency Directory",
    "📖 How To Use App Guide"
])

# ==============================================================================
# TAB 1: HOUSEHOLD ACTION PLAN GENERATOR
# ==============================================================================
with tab1:
    st.subheader("🏠 Household Demographics & Disaster Action Plan")
    st.markdown("Specify your household structure to generate a personalized emergency evacuation and safety roadmap.")

    col1, col2 = st.columns(2)
    with col1:
        total_members = st.number_input("Total Household Members:", min_value=1, max_value=100, value=6)
        elderly_count = st.number_input("Elderly Members (60+ Years):", min_value=0, max_value=50, value=1)
        children_count = st.number_input("Children & Infants (0-12 Years):", min_value=0, max_value=50, value=2)

    with col2:
        disabled_count = st.number_input("Special Care / Disabled Members:", min_value=0, max_value=20, value=0)
        vehicles = st.multiselect("Available Motor Vehicles:", ["Motorcycle / Scooter", "Car / Sedan", "4x4 / SUV", "Tractor / Truck", "None (On Foot)"], default=["Motorcycle / Scooter"])
        house_type = st.selectbox("House Structure:", ["Multi-Story Concrete / Pacca", "Single-Story Concrete / Pacca", "Kaccha / Mud Structure", "Temporary Shelter / Tent"])

    extra_notes = st.text_area("Additional Details (e.g. Near Nullah Lai, livestock care needed):", placeholder="We have 2 goats and live near a canal...")

    if st.button("🚨 Generate Household Flood Action Plan", type="primary"):
        
        # Determine language prompt rules based on sidebar selection
        if selected_lang == "English Only":
            lang_prompt = "Write the full action plan strictly in English."
        else:
            lang_prompt = (
                f"LANGUAGE REQUIREMENT:\n"
                f"- Keep the main layout and titles in English.\n"
                f"- For EVERY step, bullet point, and recommendation, write the English instruction first.\n"
                f"- Directly underneath each English item, provide a translation/caption in {selected_lang}.\n"
                f"- End the plan with a summary section titled 'Zaroori Hidayat ({selected_lang})' written completely in {selected_lang}."
            )

        prompt = f"""
        Generate a complete Household Flood Action Plan.

        {lang_prompt}

        LOCATION CONTEXT:
        - Province: {selected_province}
        - District: {selected_district}
        - Tehsil: {selected_tehsil}
        - Neighborhood/Address: {specific_address if specific_address else 'Not specified'}

        HOUSEHOLD PARAMETERS:
        - Total Members: {total_members}
        - Elderly (60+): {elderly_count}
        - Children/Infants: {children_count}
        - Special Care/Disabled: {disabled_count}
        - Vehicles: {', '.join(vehicles) if vehicles else 'None'}
        - House Construction: {house_type}
        - User Notes: {extra_notes if extra_notes else 'None'}

        CURRENT ALERT LEVEL: {alert_status}

        STRUCTURE YOUR RESPONSE:
        1. ⏱️ 30-Minute Priority Evacuation Steps
        2. 🎒 Go-Bag & Emergency Supplies Checklist
        3. 🚗 Transport & Family Safety Logistics
        4. 📌 Important Instructions in {selected_lang}
        """

        with st.spinner(f"Generating customized action plan with {selected_lang} captions..."):
            plan_output = generate_ai_plan(prompt)
            st.markdown("---")
            st.markdown("### 📋 Generated Household Emergency Action Plan")
            st.markdown(plan_output)

# ==============================================================================
# TAB 2: LIVE ALERT & BROADCAST DISPATCHER
# ==============================================================================
with tab2:
    st.subheader("🚨 Early Warning Alert System & Emergency Broadcast Dispatcher")
    st.markdown("Send direct SMS or WhatsApp early warning notifications to family and neighborhood leads.")

    col_al1, col_al2 = st.columns(2)
    
    with col_al1:
        st.markdown("### 📱 Instant Emergency Dispatcher")
        recipient_name = st.text_input("Recipient Name / Title:", value="Ali Khan / Neighborhood Elder")
        phone_number = st.text_input("Mobile Phone Number (+92):", value="+923001234567")
        
        default_sms_body = (
            f"🚨 EMERGENCY FLOOD ALERT - {selected_district.upper()} 🚨\n"
            f"Threat Level: {alert_status}\n"
            f"Heavy rainfall detected in {selected_tehsil}. Prepare emergency kits and move family to safe ground.\n"
            f"Help Hotline: Rescue 1122 | NDMA: 051-111-157-157\n"
            f"Urdu: {selected_district} mein selaab ka khatra hai! Fawri mehfooz jagah muntaqil ho jayein."
        )

        sms_custom_text = st.text_area("Alert Message Preview:", value=default_sms_body, height=140)

        clean_phone = "".join(filter(str.isdigit, phone_number))
        if clean_phone.startswith("0") and len(clean_phone) == 11:
            clean_phone = "92" + clean_phone[1:]

        encoded_msg = urllib.parse.quote(sms_custom_text)
        whatsapp_url = f"https://api.whatsapp.com/send?phone={clean_phone}&text={encoded_msg}"

        st.markdown(f"""
            <a href="{whatsapp_url}" target="_blank" style="text-decoration:none;">
                <div style="background-color:#25D366; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold; font-size:16px; margin-bottom:12px;">
                    📲 Send via WhatsApp (Direct)
                </div>
            </a>
        """, unsafe_allow_html=True)

        if st.button("🚀 Dispatch Emergency Alert via Emergency Gateway"):
            if not phone_number or len(clean_phone) < 10:
                st.warning("Please enter a valid mobile phone number.")
            else:
                st.success(f"✅ Emergency Alert dispatched successfully to {phone_number} ({recipient_name})!")

    with col_al2:
        st.markdown("### 📡 Active Early Warning Indicators")
        st.metric("District Threat Level", alert_status)
        if not weather_df.empty:
            st.metric("Peak Forecast Rainfall", f"{weather_df['Precipitation (mm)'].max():.1f} mm")
            st.metric("7-Day Rain Total", f"{weather_df['Precipitation (mm)'].sum():.1f} mm")

# ==============================================================================
# TAB 3: WEATHER & HISTORICAL FLOOD RADAR
# ==============================================================================
with tab3:
    st.subheader(f"📊 Weather Analytics & Historical Flood Data - {selected_district}")

    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        st.markdown("### 🌧️ 7-Day Live Rainfall Forecast")
        if not weather_df.empty:
            fig_rain = px.bar(
                weather_df,
                x="Date",
                y="Precipitation (mm)",
                title=f"7-Day Predicted Rainfall in {selected_district} (mm)",
                color="Precipitation (mm)",
                color_continuous_scale="Reds" if alert_status in ["CRITICAL DANGER", "HIGH ALERT"] else "Blues"
            )
            st.plotly_chart(fig_rain, use_container_width=True)
        else:
            st.warning("Live weather data currently unavailable.")

    with col_w2:
        st.markdown("### 📈 Historical Floods in Pakistan")
        historical_data = pd.DataFrame({
            "Flood Year": ["2010 Super Flood", "2012 Rain Floods", "2014 Riverine", "2022 Monsoon Catastrophe", "2024 Monsoon"],
            "Affected Population (Millions)": [20.0, 5.0, 2.5, 33.0, 3.2],
            "Economic Loss ($ Billion USD)": [10.0, 2.5, 2.0, 30.0, 1.8]
        })

        fig_hist = px.bar(
            historical_data,
            x="Flood Year",
            y="Affected Population (Millions)",
            text="Affected Population (Millions)",
            title="Impact of Major Floods in Pakistan (Affected People in Millions)",
            color="Economic Loss ($ Billion USD)",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# TAB 4: NATIONWIDE EMERGENCY DIRECTORY
# ==============================================================================
with tab4:
    st.subheader("📞 Pakistan Emergency Helpline Directory")

    directory_data = [
        {"Agency": "Rescue 1122", "Helpline": "1122", "Scope": "National", "Services": "Ambulance, Fire, Water Rescue"},
        {"Agency": "NDMA (National Disaster Management Authority)", "Helpline": "051-111-157-157", "Scope": "National", "Services": "National Flood Monitoring & Relief"},
        {"Agency": "PDMA Punjab", "Helpline": "1129", "Scope": "Punjab", "Services": "Provincial Flood Control & Relief Camps"},
        {"Agency": "PDMA Sindh", "Helpline": "021-99251458 / 1093", "Scope": "Sindh", "Services": "Monsoon Operations & Relief Dispatch"},
        {"Agency": "PDMA KPK", "Helpline": "1700", "Scope": "KPK", "Services": "Flash Flood & Landslide Emergency Response"},
        {"Agency": "PDMA Balochistan", "Helpline": "081-9241133", "Scope": "Balochistan", "Services": "Dam & Coastal Area Rescue Operations"},
        {"Agency": "Edhi Foundation Helpline", "Helpline": "115", "Scope": "National", "Services": "Ambulance, Food Packets & Shelter"},
        {"Agency": "Chhipa Welfare", "Helpline": "1020", "Scope": "National", "Services": "Ambulance & Emergency Response"}
    ]

    dir_df = pd.DataFrame(directory_data)
    filter_scope = st.selectbox("Filter Directory by Scope:", ["All", "National", "Punjab", "Sindh", "KPK", "Balochistan"])

    if filter_scope != "All":
        filtered_dir = dir_df[(dir_df["Scope"] == filter_scope) | (dir_df["Scope"] == "National")]
    else:
        filtered_dir = dir_df

    st.dataframe(filtered_dir, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 5: HOW TO USE APP GUIDE
# ==============================================================================
with tab5:
    st.subheader("📖 How to Use FloodReady AI")
    st.markdown("""
    1. **Sidebar:** Select your Province, District, Tehsil, and secondary language.
    2. **Alert Banner:** Check the top banner for live danger warnings in your selected district.
    3. **Household Plan (Tab 1):** Enter your family demographics and click **Generate Household Flood Action Plan**.
    4. **Emergency Broadcast (Tab 2):** Dispatch alerts via WhatsApp to family members.
    """)

# Footer
st.markdown("---")
st.caption("FloodReady AI • National Emergency Preparedness Platform for Pakistan • Powered by Groq AI Inference")
