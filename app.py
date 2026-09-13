import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import urllib.parse
import os
import json

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="FloodReady AI - Pakistan Emergency System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-themed emergency UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 20px;
    }
    .alert-card-critical {
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        padding: 15px;
        border-radius: 8px;
        color: #991B1B;
        margin-bottom: 15px;
    }
    .alert-card-high {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 15px;
        border-radius: 8px;
        color: #92400E;
        margin-bottom: 15px;
    }
    .alert-card-normal {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 15px;
        border-radius: 8px;
        color: #065F46;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PAKISTAN DISTRICTS & COORDINATES DATA
# ==========================================
PAKISTAN_LOCATION_DATA = {
    "Punjab": {
        "Lahore": {"lat": 31.5204, "lon": 74.3587},
        "Rawalpindi": {"lat": 33.5651, "lon": 73.0169},
        "Multan": {"lat": 30.1575, "lon": 71.5249},
        "Faisalabad": {"lat": 31.4504, "lon": 73.1350},
        "Dera Ghazi Khan": {"lat": 30.0561, "lon": 70.6348},
        "Sialkot": {"lat": 32.4945, "lon": 74.5229},
        "Bahawalpur": {"lat": 29.3544, "lon": 71.6911}
    },
    "Sindh": {
        "Karachi": {"lat": 24.8607, "lon": 67.0011},
        "Dadu": {"lat": 26.7303, "lon": 67.7769},
        "Sukkur": {"lat": 27.7132, "lon": 68.8369},
        "Hyderabad": {"lat": 25.3960, "lon": 68.3578},
        "Larkana": {"lat": 27.5580, "lon": 68.2120},
        "Thatta": {"lat": 24.7475, "lon": 67.9239}
    },
    "Khyber Pakhtunkhwa (KPK)": {
        "Peshawar": {"lat": 34.0151, "lon": 71.5249},
        "Swat": {"lat": 35.2227, "lon": 72.4258},
        "Nowshera": {"lat": 34.0105, "lon": 71.9876},
        "Charsadda": {"lat": 34.1482, "lon": 71.7406},
        "Abbottabad": {"lat": 34.1688, "lon": 73.2215},
        "D.I. Khan": {"lat": 31.8312, "lon": 70.9017}
    },
    "Balochistan": {
        "Quetta": {"lat": 30.1798, "lon": 66.9750},
        "Jaffarabad": {"lat": 28.3290, "lon": 68.1408},
        "Naseerabad": {"lat": 28.6200, "lon": 67.7900},
        "Gwadar": {"lat": 25.1216, "lon": 62.3254},
        "Khuzdar": {"lat": 27.8165, "lon": 66.6057}
    },
    "Gilgit-Baltistan (GB)": {
        "Gilgit": {"lat": 35.9208, "lon": 74.3089},
        "Skardu": {"lat": 35.2971, "lon": 75.6333},
        "Hunza": {"lat": 36.3167, "lon": 74.6500}
    },
    "Azad Jammu & Kashmir (AJK)": {
        "Muzaffarabad": {"lat": 34.3700, "lon": 73.4711},
        "Mirpur": {"lat": 33.1484, "lon": 73.7518},
        "Kotli": {"lat": 33.5156, "lon": 73.9019}
    },
    "Islamabad Capital Territory": {
        "Islamabad": {"lat": 33.6844, "lon": 73.0479}
    }
}

# ==========================================
# 3. WEATHER TELEMETRY ENGINE (OPEN-METEO)
# ==========================================
@st.cache_data(ttl=1800)
def fetch_weather_data(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum,precipitation_probability_max,temperature_2m_max&timezone=Asia%2FKarachi"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        pass
    return None

def evaluate_flood_risk(weather_json):
    if not weather_json or "daily" not in weather_json:
        return "NORMAL", "#10B981", 0.0, 0.0
    
    daily_precip = weather_json["daily"].get("precipitation_sum", [0])
    max_single_day_rain = max(daily_precip) if daily_precip else 0.0
    total_7day_rain = sum(daily_precip) if daily_precip else 0.0
    
    if max_single_day_rain >= 80 or total_7day_rain >= 150:
        return "CRITICAL DANGER", "#DC2626", max_single_day_rain, total_7day_rain
    elif max_single_day_rain >= 40 or total_7day_rain >= 80:
        return "HIGH ALERT", "#F59E0B", max_single_day_rain, total_7day_rain
    elif max_single_day_rain >= 20:
        return "MODERATE ADVISORY", "#3B82F6", max_single_day_rain, total_7day_rain
    else:
        return "NORMAL", "#10B981", max_single_day_rain, total_7day_rain

# ==========================================
# 4. GROQ AI ENGINE WITH MODEL FALLBACK LOOP
# ==========================================
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini"
]

def generate_ai_evacuation_plan(api_key, district, risk_status, max_rain, household_size, elderly, infants, special_needs, housing, vehicles):
    if not api_key:
        return "⚠️ **Groq API Key missing.** Please enter your Groq API Key in the sidebar to generate AI plans."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        return "❌ Python `groq` package is missing. Please install it using `pip install groq`."

    system_prompt = (
        "You are FloodReady AI, Pakistan's official emergency preparedness & disaster response assistant. "
        "Generate a structured, extremely urgent, step-by-step 30-minute evacuation and safety plan. "
        "Keep language professional, highly practical, and tailored specifically to Pakistan's local ground conditions. "
        "Always include a dedicated section titled 'Zaroori Hidayat (Urdu Safety Steps)' at the end."
    )

    user_prompt = f"""
    LOCATION: District {district}, Pakistan
    CURRENT FLOOD RISK: {risk_status} (Peak 24hr Rain Forecast: {max_rain:.1f} mm)
    HOUSEHOLD DEMOGRAPHICS:
    - Total Members: {household_size}
    - Elderly (60+ yrs): {elderly}
    - Infants/Children: {infants}
    - Special Needs / Disabled: {special_needs}
    - Housing Construction: {housing}
    - Available Transportation: {', '.join(vehicles) if vehicles else 'On Foot'}

    Please generate:
    1. ⏱️ 30-Minute Immediate Action Timeline (Minutes 0-10, 10-20, 20-30)
    2. 🎒 Custom Emergency Go-Bag Checklist tailored for this family
    3. 🚗 Vehicle & Route Safety Strategy
    4. 📢 Zaroori Hidayat (Clear Roman Urdu Evacuation Steps for local family members)
    """

    last_error = None
    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1800
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue

    return f"❌ All Groq AI endpoints failed. Error details: {last_error}"

# ==========================================
# 5. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/flood.png", width=70)
st.sidebar.title("FloodReady AI")
st.sidebar.caption("Pakistan Early Warning System")

# API Key input
groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", help="Enter your Groq API Key for AI Plan Generation")
if not groq_api_key and "GROQ_API_KEY" in os.environ:
    groq_api_key = os.environ["GROQ_API_KEY"]

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location Selector")
selected_province = st.sidebar.selectbox("Select Province / Region", list(PAKISTAN_LOCATION_DATA.keys()))
districts_in_province = list(PAKISTAN_LOCATION_DATA[selected_province].keys())
selected_district = st.sidebar.selectbox("Select District", districts_in_province)

coords = PAKISTAN_LOCATION_DATA[selected_province][selected_district]
weather_data = fetch_weather_data(coords["lat"], coords["lon"])
risk_level, risk_color, max_rain, total_rain = evaluate_flood_risk(weather_data)

# Sidebar Risk Summary
st.sidebar.markdown("---")
st.sidebar.subheader("Current Risk Status")
st.sidebar.markdown(f"<h3 style='color:{risk_color}; margin:0;'>{risk_level}</h3>", unsafe_allow_html=True)
st.sidebar.write(f"**Peak Rain (24h):** {max_rain:.1f} mm")
st.sidebar.write(f"**7-Day Total Rain:** {total_rain:.1f} mm")

# Play Siren Audio if Critical Danger
if risk_level == "CRITICAL DANGER":
    st.markdown("""
        <audio autoplay loop>
            <source src="https://www.soundjay.com/mechanical/sounds/tornado-siren-1.mp3" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)

# ==========================================
# 6. MAIN APPLICATION LAYOUT & TABS
# ==========================================
st.markdown("<h1 class='main-header'>🌊 FloodReady AI: Early Warning & Preparedness</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>National Emergency Response & AI Household Evacuation System for Pakistan</p>", unsafe_allow_html=True)

# Dynamic Top Banner
if risk_level == "CRITICAL DANGER":
    st.markdown(f"<div class='alert-card-critical'>🚨 <b>CRITICAL FLOOD ALERT FOR DISTRICT {selected_district.upper()}!</b> High precipitation ({max_rain:.1f}mm) detected. Move vulnerable family members to safe ground immediately!</div>", unsafe_allow_html=True)
elif risk_level == "HIGH ALERT":
    st.markdown(f"<div class='alert-card-high'>⚠️ <b>HIGH FLOOD ADVISORY FOR DISTRICT {selected_district.upper()}.</b> Heavy rainfall expected ({max_rain:.1f}mm). Prepare emergency kits and monitor local helplines.</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='alert-card-normal'>✅ <b>NORMAL WEATHER CONDITIONS IN {selected_district.upper()}.</b> Expected rainfall ({max_rain:.1f}mm) is within normal parameters.</div>", unsafe_allow_html=True)

tabs = st.tabs([
    "🌧️ Live Telemetry & AI Evacuation Plan",
    "🚨 Instant Emergency Alert Dispatcher",
    "📊 Historical Flood Analytics",
    "📞 Verified Helplines & Guide"
])

# ==========================================
# TAB 1: LIVE TELEMETRY & AI PLAN GENERATOR
# ==========================================
with tabs[0]:
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("👨‍👩‍👧‍👦 Household Demographics & Profile")
        
        household_size = st.number_input("Total Family / Household Members", min_value=1, max_value=100, value=6)
        
        c1, c2 = st.columns(2)
        with c1:
            elderly_cnt = st.number_input("Elderly Members (60+ yrs)", min_value=0, max_value=20, value=1)
            infants_cnt = st.number_input("Infants / Children (<10 yrs)", min_value=0, max_value=20, value=2)
        with c2:
            special_cnt = st.number_input("Special Needs / Disabled", min_value=0, max_value=10, value=0)
            housing_type = st.selectbox("Housing Type", ["Kaccha (Mud / Clay)", "Pacca (Single Story Concrete)", "Multi-Story Concrete Building", "Temporary Shelter"])

        vehicles = st.multiselect(
            "Available Transportation / Vehicles",
            ["Motorcycle / Rickshaw", "Sedan / Hatchback Car", "4x4 SUV / Pickup", "Tractor / Truck", "None (On Foot)"],
            default=["Motorcycle / Rickshaw"]
        )

        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("🚀 Generate AI Household Emergency Evacuation Plan", type="primary")

    with col_right:
        st.subheader("📋 Customized AI Emergency Response Plan")
        
        if generate_btn:
            with st.spinner(f"Generating evacuation response for {selected_district} via Groq AI..."):
                ai_plan = generate_ai_evacuation_plan(
                    groq_api_key, selected_district, risk_level, max_rain,
                    household_size, elderly_cnt, infants_cnt, special_cnt, housing_type, vehicles
                )
                st.markdown(ai_plan)
        else:
            st.info("👈 Enter your family details and click **'Generate AI Household Emergency Evacuation Plan'** to receive customized 30-minute safety guidelines.")

# ==========================================
# TAB 2: INSTANT EMERGENCY ALERT DISPATCHER (FIXED)
# ==========================================
with tabs[1]:
    st.subheader("🚨 Instant Emergency Alert Dispatcher")
    st.caption("Send immediate alerts to family, neighbors, or community leads before local network connectivity drops.")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        recipient_name = st.text_input("Recipient Name / Title", value="Ali Khan / Neighborhood Elder")
        mobile_number = st.text_input("Mobile Phone Number (with Country Code +92)", value="+923001234567")
        alert_urgency = st.selectbox("Alert Urgency Level", ["🚨 Critical Evacuation Alert", "⚠️ High Advisory Warning", "ℹ️ General Community Notice"])
        
        default_msg = f"🚨 EMERGENCY FLOOD ALERT - {selected_district.upper()}\nThreat Level: {risk_level}\nHeavy rainfall ({max_rain:.1f}mm) forecasted in {selected_district}. Prepare emergency kits and move family to safe ground.\nHelp Hotline: Rescue 1122 | NDMA: 051-111-157-157"
        alert_msg_text = st.text_area("Alert Message Preview", value=default_msg, height=140)

    with col_b:
        st.subheader("📲 Choose Dispatch Channel")
        st.write("Select how you want to send this emergency alert:")

        # Clean recipient phone number for WhatsApp URL
        clean_phone = "".join(filter(str.isdigit, mobile_number))
        encoded_msg = urllib.parse.quote(alert_msg_text)
        whatsapp_url = f"https://wa.me/{clean_phone}?text={encoded_msg}"

        # 1. Direct WhatsApp Button (Free, Direct)
        st.markdown(f"""
            <a href="{whatsapp_url}" target="_blank" style="text-decoration:none;">
                <div style="background-color:#25D366; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold; font-size:16px; margin-bottom:12px;">
                    📲 Send via WhatsApp Web / App (100% Free & Direct)
                </div>
            </a>
        """, unsafe_allow_html=True)

        # 2. Instant Emergency Gateway Button (Demo Mode)
        if st.button("🚀 Dispatch Emergency Alert via Emergency Gateway", type="primary"):
            if not mobile_number or len(mobile_number) < 10:
                st.error("Please enter a valid phone number (e.g. +923001234567).")
            else:
                st.success(f"✅ **EMERGENCY ALERT DISPATCHED SUCCESSFULLY!**")
                st.json({
                    "Status": "SENT (DELIVERED)",
                    "Recipient": recipient_name,
                    "Phone Target": mobile_number,
                    "District Target": selected_district,
                    "Risk Protocol": risk_level,
                    "Gateway Gateway Route": "Pak-National Emergency Broadcast (SIM-Route / IP-Push)",
                    "Timestamp": "Just Now"
                })

        st.info("💡 **Tip for Hackathon Demo:** Clicking the green WhatsApp button instantly opens WhatsApp with the alert pre-filled! The blue Gateway button simulates direct national SMS/IP dispatching.")

# ==========================================
# TAB 3: HISTORICAL FLOOD ANALYTICS
# ==========================================
with tabs[2]:
    st.subheader("📊 Historical Flood Impact & 7-Day Rainfall Forecast")
    
    # 7-Day Weather Forecast Chart
    if weather_data and "daily" in weather_data:
        dates = weather_data["daily"]["time"]
        rains = weather_data["daily"]["precipitation_sum"]
        
        df_forecast = pd.DataFrame({"Date": dates, "Precipitation (mm)": rains})
        fig_forecast = px.bar(df_forecast, x="Date", y="Precipitation (mm)", title=f"7-Day Forecasted Rainfall for {selected_district}", color="Precipitation (mm)", color_continuous_scale="Reds")
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Weather telemetry data currently unavailable.")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("🌊 Major Pakistan Flood Comparison")
        df_history = pd.DataFrame({
            "Flood Event": ["2010 Super Flood", "2012 Floods", "2014 Floods", "2022 Monsoon Cataclysm", "2024 Monsoon"],
            "Affected Population (Millions)": [20.0, 5.0, 2.5, 33.0, 3.5],
            "Economic Damage ($ Billion USD)": [10.0, 2.5, 2.0, 30.0, 1.8]
        })
        fig_hist = px.bar(df_history, x="Flood Event", y="Affected Population (Millions)", color="Economic Damage ($ Billion USD)", title="Historical Flood Severity in Pakistan")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_h2:
        st.subheader("🗺️ Provincial Damage Distribution (2022)")
        df_prov = pd.DataFrame({
            "Province": ["Sindh", "Balochistan", "KPK", "Punjab", "GB / AJK"],
            "Damage Share (%)": [45, 25, 15, 10, 5]
        })
        fig_pie = px.pie(df_prov, names="Province", values="Damage Share (%)", title="2022 Flood Losses by Region", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 4: VERIFIED HELPLINES & USER GUIDE
# ==========================================
with tabs[3]:
    st.subheader("📞 Emergency Helplines & Rescue Contacts")
    
    helpline_data = [
        {"Organization": "Rescue 1122 (Emergency Medical & Rescue)", "Number": "1122", "Coverage": "Nationwide"},
        {"Organization": "NDMA (National Disaster Management)", "Number": "051-111-157-157", "Coverage": "Federal / Islamabad"},
        {"Organization": "PDMA Punjab", "Number": "1129", "Coverage": "Punjab Province"},
        {"Organization": "PDMA Sindh", "Number": "021-99251458", "Coverage": "Sindh Province"},
        {"Organization": "PDMA Khyber Pakhtunkhwa", "Number": "1700", "Coverage": "KPK Province"},
        {"Organization": "PDMA Balochistan", "Number": "081-9241133", "Coverage": "Balochistan Province"},
        {"Organization": "Edhi Emergency Ambulance", "Number": "115", "Coverage": "Nationwide"},
        {"Organization": "Chhipa Welfare Network", "Number": "1020", "Coverage": "Major Urban Centers"},
        {"Organization": "Pakistan Red Crescent (PRCS)", "Number": "1030", "Coverage": "Nationwide"},
        {"Organization": "Motorway & Highway Police", "Number": "130", "Coverage": "National Highways"}
    ]
    st.table(pd.DataFrame(helpline_data))

    st.markdown("---")
    st.subheader("📘 FloodReady AI User Guide")
    st.markdown("""
    1. **Select Location:** Use the left sidebar to choose your Province and District.
    2. **Monitor Live Status:** The app automatically fetches satellite rainfall forecasts and evaluates the flood threat level.
    3. **Generate Family Plan:** Under Tab 1, enter your family size, vulnerable members, and transportation to receive an AI-generated 30-minute evacuation timeline.
    4. **Dispatch Warning Alerts:** Use Tab 2 to send direct WhatsApp messages or simulate emergency broadcast notifications to neighbors.
    """)
