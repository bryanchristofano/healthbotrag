import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from pathlib import Path
from PIL import Image
from streamlit_option_menu import option_menu
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
doctorpic = current_dir / "healthbot-pic.png"

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Doctor",
    page_icon=":brain:",
    layout="centered",
    initial_sidebar_state="expanded",
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)

# Initialize Firebase connection using REST API
import requests

@st.cache_resource
def initialize_firebase():
    """Initialize Firebase connection using REST API"""
    try:
        # Test connection to Firebase
        test_url = "https://healthbot-ceb8d-default-rtdb.asia-southeast1.firebasedatabase.app/.json"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Firebase connection failed: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error connecting to Firebase: {e}")
        return False

# Load Gemini-Pro model
def gemini_pro():
    model = gen_ai.GenerativeModel('gemini-1.5-flash')
    return model

# Load Gemini Vision model
def gemini_vision():
    model = gen_ai.GenerativeModel('gemini-1.5-flash')
    return model

# Load PDFs from folder
def load_pdfs_from_folder(folder_path):
    knowledge_base = []
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        text = "\n".join([page.get_text("text") for page in doc])
        knowledge_base.append(text)
        doc.close()
    
    return knowledge_base

# PERBAIKAN UTAMA: Fungsi untuk membaca struktur JSON dengan multi-person
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_bmp_history():
    """Fetch BPM history from Firebase using REST API - supports multi-person structure"""
    try:
        # Gunakan REST API konsisten dengan initialize_firebase()
        url = "https://healthbot-ceb8d-default-rtdb.asia-southeast1.firebasedatabase.app/bpm_history.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            bmp_data = response.json()
            
            if bmp_data:
                # Handle struktur JSON baru dengan multi-person
                if isinstance(bmp_data, dict):
                    # Check if this is the new structure with person names as keys
                    person_keys = list(bmp_data.keys())
                    if person_keys and isinstance(bmp_data[person_keys[0]], dict):
                        # New structure: {"glenn": {"timestamp": bpm}, "bryan": {...}}
                        return parse_multi_person_bmp_data(bmp_data)
                    else:
                        # Old structure: {"timestamp": bmp}
                        return parse_single_bmp_data(bmp_data)
                elif isinstance(bmp_data, list):
                    # Array structure: [{"timestamp": "...", "bpm": ...}]
                    return parse_array_bmp_data(bmp_data)
            
            return []
        else:
            st.error(f"Error fetching BPM data: HTTP {response.status_code}")
            return []
            
    except requests.exceptions.Timeout:
        st.error("Timeout saat mengambil data BPM dari Firebase")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error koneksi ke Firebase: {e}")
        return []
    except Exception as e:
        st.error(f"Error unexpected saat mengambil data BPM: {e}")
        return []

def parse_multi_person_bmp_data(bmp_data):
    """Parse BPM data with multi-person structure"""
    bmp_list = []
    
    for person_name, person_data in bmp_data.items():
        if isinstance(person_data, dict):
            for timestamp, bmp_value in person_data.items():
                try:
                    # Parse timestamp format: YYYY-MM-DD_HH:MM:SS
                    dt = datetime.strptime(timestamp, "%Y-%m-%d_%H:%M:%S")
                    bmp_list.append({
                        'timestamp': timestamp,
                        'datetime': dt,
                        'bpm': bmp_value,
                        'person': person_name
                    })
                except ValueError as e:
                    print(f"Error parsing timestamp {timestamp}: {e}")
                    continue
                except (TypeError, KeyError) as e:
                    print(f"Error parsing BPM data for {person_name}: {e}")
                    continue
    
    # Sort by datetime
    bmp_list.sort(key=lambda x: x['datetime'])
    return bmp_list

def parse_single_bmp_data(bmp_data):
    """Parse BPM data with single structure (legacy)"""
    bmp_list = []
    for timestamp, bmp_value in bmp_data.items():
        try:
            # Parse timestamp format: YYYY-MM-DD_HH:MM:SS
            dt = datetime.strptime(timestamp, "%Y-%m-%d_%H:%M:%S")
            bmp_list.append({
                'timestamp': timestamp,
                'datetime': dt,
                'bpm': bmp_value,
                'person': 'unknown'
            })
        except ValueError:
            continue
    
    bmp_list.sort(key=lambda x: x['datetime'])
    return bmp_list

def parse_array_bmp_data(bmp_data):
    """Parse BPM data with array structure"""
    bmp_list = []
    for item in bmp_data:
        try:
            # Format timestamp: "2025-05-20 15:56:14"
            dt = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")
            bmp_list.append({
                'timestamp': item['timestamp'],
                'datetime': dt,
                'bpm': item['bpm'],
                'person': item.get('person', 'unknown')
            })
        except (ValueError, KeyError) as e:
            print(f"Skipping invalid data: {item}, Error: {e}")
            continue
    
    bmp_list.sort(key=lambda x: x['datetime'])
    return bmp_list

# Analyze BPM data - DIPERBAIKI untuk mendukung multi-person
def analyze_bmp_data(bmp_data, selected_person=None):
    """Analyze BPM data and generate insights - supports multi-person analysis"""
    if not bmp_data:
        return "Tidak ada data BPM yang tersedia untuk dianalisis."
    
    # Filter data berdasarkan person jika dipilih
    if selected_person and selected_person != 'all':
        filtered_data = [item for item in bmp_data if item.get('person') == selected_person]
        if not filtered_data:
            return f"Tidak ada data BPM untuk {selected_person}."
        analysis_data = filtered_data
        person_text = f" untuk {selected_person.title()}"
    else:
        analysis_data = bmp_data
        person_text = " (semua orang)" if len(set(item.get('person', 'unknown') for item in bmp_data)) > 1 else ""
    
    bmp_values = [item['bpm'] for item in analysis_data]
    timestamps = [item['timestamp'] for item in analysis_data]
    
    # Basic statistics
    avg_bpm = sum(bmp_values) / len(bmp_values)
    max_bpm = max(bmp_values)
    min_bpm = min(bmp_values)
    latest_bmp = bmp_values[-1] if bmp_values else 0
    
    # Health assessment
    def assess_bpm(bpm):
        if bpm < 60:
            return "Bradikardia (detak jantung lambat)"
        elif 60 <= bpm <= 100:
            return "Normal"
        elif 100 < bpm <= 120:
            return "Sedikit tinggi"
        else:
            return "Takikardia (detak jantung cepat)"
    
    # Get unique persons for summary
    persons = list(set(item.get('person', 'unknown') for item in bmp_data))
    
    analysis = f"""
    ANALISIS DATA BPM{person_text}:
    
    📊 STATISTIK UMUM:
    • Total data: {len(analysis_data)} pengukuran
    • BPM rata-rata: {avg_bpm:.1f} bpm
    • BPM tertinggi: {max_bpm} bpm
    • BPM terendah: {min_bpm} bpm
    • BPM terakhir: {latest_bmp} bpm
    """
    
    if len(persons) > 1 and selected_person == 'all':
        analysis += f"\n    • Jumlah orang dipantau: {len(persons)} ({', '.join(persons)})"
    
    analysis += f"""
    
    🕐 PERIODE DATA:
    • Data pertama: {timestamps[0] if timestamps else 'Tidak ada'}
    • Data terakhir: {timestamps[-1] if timestamps else 'Tidak ada'}
    
    ❤️ PENILAIAN KESEHATAN:
    • Status BPM rata-rata: {assess_bpm(avg_bpm)}
    • Status BPM terakhir: {assess_bpm(latest_bmp)}
    
    📈 TREN:
    """
    
    # Trend analysis
    if len(bmp_values) >= 3:
        recent_avg = sum(bmp_values[-3:]) / 3
        older_avg = sum(bmp_values[:3]) / 3 if len(bmp_values) >= 6 else avg_bpm
        
        if recent_avg > older_avg + 5:
            analysis += "• Tren meningkat - BPM cenderung naik dalam pengukuran terbaru\n"
        elif recent_avg < older_avg - 5:
            analysis += "• Tren menurun - BPM cenderung turun dalam pengukuran terbaru\n"
        else:
            analysis += "• Tren stabil - BPM relatif konsisten\n"
    
    # Recommendations
    analysis += f"""
    
    💡 REKOMENDASI:
    """
    
    if avg_bpm < 60:
        analysis += "• Konsultasi dengan dokter mengenai bradikardia\n• Monitor aktivitas dan gejala seperti pusing atau kelelahan\n"
    elif avg_bpm > 100:
        analysis += "• Pertimbangkan faktor stress, kafein, atau aktivitas fisik\n• Konsultasi dengan dokter jika takikardia persisten\n"
    else:
        analysis += "• BPM dalam rentang normal, pertahankan gaya hidup sehat\n• Lanjutkan monitoring rutin\n"
    
    analysis += "• Catat gejala yang mungkin terkait dengan perubahan BPM\n• Pertimbangkan faktor lingkungan dan aktivitas saat pengukuran"
    
    # Per-person summary jika multi-person
    if len(persons) > 1 and selected_person in ['all', None]:
        analysis += "\n\n👥 RINGKASAN PER ORANG:\n"
        for person in persons:
            person_data = [item for item in bmp_data if item.get('person') == person]
            if person_data:
                person_avg = sum(item['bpm'] for item in person_data) / len(person_data)
                analysis += f"• {person.title()}: {len(person_data)} data, rata-rata {person_avg:.1f} bpm ({assess_bpm(person_avg)})\n"
    
    return analysis

# Generate response using the Gemini model and chat history - DIPERBAIKI
def generate_response(input_text, knowledge_base, chat_history, bmp_data=None):
    knowledge_summary = " ".join(knowledge_base)
    
    # Check if user asks about BPM history
    bmp_keywords = ['bpm', 'detak jantung', 'heart rate', 'jantung', 'beat', 'pulse']
    history_keywords = ['history', 'histori', 'riwayat', 'data', 'analisis', 'analysis']
    
    is_bmp_query = any(keyword in input_text.lower() for keyword in bmp_keywords)
    is_history_query = any(keyword in input_text.lower() for keyword in history_keywords)
    
    bmp_context = ""
    if (is_bmp_query or is_history_query) and bmp_data:
        # Check if user asks for specific person
        persons = list(set(item.get('person', 'unknown') for item in bmp_data))
        selected_person = None
        
        for person in persons:
            if person.lower() in input_text.lower():
                selected_person = person
                break
        
        bmp_analysis = analyze_bmp_data(bmp_data, selected_person)
        bmp_context = f"\n\nDATA BPM PASIEN:\n{bmp_analysis}\n"

    # Format riwayat percakapan untuk Gemini
    history_text = "\n".join([f"{role}: {text}" for role, text in chat_history])

    # Gabungkan knowledge base, BPM data dan riwayat percakapan ke dalam prompt
    full_prompt = f"{knowledge_summary}{bmp_context}\n\nRiwayat Percakapan:\n{history_text}\n\nUser: {input_text}\nAssistant:"

    if input_text.lower() in ["siapa namamu", "who are you", "siapa kamu", "kamu siapa", "siapa anda", "siapa kamu?", "siapa namamu?", "who are you?", "kamu siapa?", "siapa anda?", "kamu adalah apa", "kamu adalah apa?"]:
        return "Saya adalah HealthBot buatan Glenn dan Bryan berdasarkan knowledge base yang diberikan oleh mereka. Saya juga dapat menganalisis data BPM dari perangkat monitoring multi-user Anda. Terima kasih Glenn dan Bryan."

    response = get_gemini_response(full_prompt)
    
    return response

# Function to get response from Gemini
def get_gemini_response(question):
    response = model.generate_content(question)
    return response.text

# Get response from Gemini Vision model, including knowledge base
def gemini_vision_response(model, prompt, image, knowledge_base):
    knowledge_summary = " ".join(knowledge_base)
    full_prompt = f"{knowledge_summary}\n{prompt}"
    
    response = model.generate_content([full_prompt, image])
    return response.text

# Create BMP visualization - DIPERBAIKI untuk multi-person
def create_bmp_chart(bmp_data, selected_person=None):
    """Create BPM trend chart - supports multi-person visualization"""
    if not bmp_data:
        return None
    
    # Filter data jika person dipilih
    if selected_person and selected_person != 'all':
        filtered_data = [item for item in bmp_data if item.get('person') == selected_person]
        if not filtered_data:
            return None
        plot_data = filtered_data
        title_suffix = f" - {selected_person.title()}"
    else:
        plot_data = bmp_data
        title_suffix = ""
    
    df = pd.DataFrame(plot_data)
    
    fig = go.Figure()
    
    # Jika ada multiple persons, buat line untuk setiap orang
    persons = list(set(item.get('person', 'unknown') for item in plot_data))
    
    if len(persons) > 1 and (selected_person is None or selected_person == 'all'):
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, person in enumerate(persons):
            person_data = [item for item in plot_data if item.get('person') == person]
            person_df = pd.DataFrame(person_data)
            
            fig.add_trace(go.Scatter(
                x=person_df['datetime'],
                y=person_df['bpm'],
                mode='lines+markers',
                name=f'{person.title()}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
    else:
        # Single person atau filtered data
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['bpm'],
            mode='lines+markers',
            name='BPM',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
    
    # Add normal range area
    fig.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Normal Min (60)")
    fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Normal Max (100)")
    fig.add_hrect(y0=60, y1=100, fillcolor="green", opacity=0.1, annotation_text="Normal Range")
    
    fig.update_layout(
        title=f"Tren BPM (Beats Per Minute){title_suffix}",
        xaxis_title="Waktu",
        yaxis_title="BPM",
        hovermode='x unified'
    )
    
    return fig

# Load knowledge base from PDFs
knowledge_base = load_pdfs_from_folder("docs")

# Initialize Firebase
firebase_initialized = initialize_firebase()

# Initialize coin count
if 'coins' not in st.session_state:
    st.session_state['coins'] = 10  # Give 10 coins initially

# Function to deduct coins and handle redirection when coins are 0
def deduct_coin():
    if st.session_state['coins'] > 0:
        st.session_state['coins'] -= 1
    if st.session_state['coins'] == 0:
        st.warning("You have run out of coins! Redirecting in 5 seconds...")
        st.markdown(f'<meta http-equiv="refresh" content="5; url=https://www.youtube.com/" />', unsafe_allow_html=True)
        st.stop()  # Prevent further actions when coins are 0

# Check if coins are already 0 and redirect user on page load
if st.session_state['coins'] == 0:
    st.markdown(f'<meta http-equiv="refresh" content="5; url=https://www.youtube.com/" />', unsafe_allow_html=True)
    st.warning("You have run out of coins! Redirecting in 5 seconds...")
    st.stop()

# Sidebar menu for choosing between Manual Guide, ChatBot, Image Captioning, and BPM Monitor
with st.sidebar:
    user_picked = option_menu(
        "HealthBot",
        ["Manual Guide", "Chat Doctor", "Image Solutions", "BPM Monitor"],
        menu_icon="robot",
        icons=["info-circle", "chat-dots-fill", "image-fill", "heart-pulse-fill"],
        default_index=0
    )

# Doctor profile picture
doctorPic = Image.open(doctorpic)
doctorPic = doctorPic.resize((200, 200))

# Display remaining coins and Firebase status
st.sidebar.write(f"Remaining Coins: {st.session_state['coins']}")
if firebase_initialized:
    st.sidebar.success("🟢 Firebase Connected")
else:
    st.sidebar.error("🔴 Firebase Disconnected")

if user_picked == 'Manual Guide':
    st.title("Manual Guide for HealthBot")
    
    video_file = open("healthbot-manual.mp4", "rb")
    video_bytes = video_file.read()

    st.video(video_bytes)

    st.header("How to Use Chat Doctor")
    st.write("""
        1. Go to the 'Chat Doctor' section from the sidebar.
        2. Enter your question in the chat box.
        3. Each question deducts 1 coin. Ensure you have sufficient coins.
        4. Your question will be answered based on the knowledge base of medical information.
        5. Ask about BPM history to get analysis of your heart rate data.
        6. You can ask for specific person's BPM data (e.g., "show glenn's BPM data").
        7. When your coins run out, you will be redirected to the subscription page.
    """)

    st.header("How to Use Image Solutions")
    st.write("""
        1. Go to the 'Image Solutions' section from the sidebar.
        2. Upload an image related to your health query.
        3. Enter a prompt describing what you want to know about the image.
        4. The system will generate a response using the Gemini Vision model.
        5. Each image submission deducts 1 coin.
    """)
    
    st.header("How to Use BPM Monitor")
    st.write("""
        1. Go to the 'BPM Monitor' section from the sidebar.
        2. View real-time BPM data from multiple ESP32 devices.
        3. Filter data by person or view all together.
        4. Analyze trends and get health insights.
        5. Export data for medical consultations.
    """)

elif user_picked == 'Chat Doctor':
    model = gemini_pro()

    # Get BPM data for context
    bmp_data = get_bmp_history() if firebase_initialized else []

    # Initialize chat session if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Doctor profile display
    st.markdown("""
        <style>
        .doctor-header {
            display: flex;
            justify-content: flex-start;
            align-items: center;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="small")
    with col1:
        st.image(doctorPic)

    with col2:
        # Get unique persons from BPM data
        persons = list(set(item.get('person', 'unknown') for item in bmp_data)) if bmp_data else []
        persons_text = f" ({', '.join(persons)})" if len(persons) > 1 else ""
        
        st.markdown(f"""
            <div class="doctor-header">
                <h1>Dr. Healthbot</h1>
            </div>
            <p>I specialize in skin, genital health, and multi-user BPM analysis</p>
            <p><b>Remaining Coins:</b> {st.session_state['coins']}</p>
            <p><b>BPM Data:</b> {len(bmp_data)} records available{persons_text}</p>
        """, unsafe_allow_html=True)

    # Display the chat history
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Something... (Try asking about specific person's BPM: 'show glenn's data')")
    if user_prompt:
        deduct_coin()  # Deduct 1 coin for every prompt
        st.session_state.chat_history.append(("user", user_prompt))
        st.chat_message("user").markdown(user_prompt)

        gemini_response = generate_response(user_prompt, knowledge_base, st.session_state.chat_history, bmp_data)

        st.session_state.chat_history.append(("assistant", gemini_response))

        with st.chat_message("assistant"):
            st.markdown(gemini_response)

elif user_picked == 'Image Solutions':
    model = gemini_vision()
    
    column1, column2 = st.columns([1, 2], gap="small")
    with column1:
        st.image(doctorPic)

    with column2:
        st.markdown(f"""
            <div class="doctor-header">
                <h1>Visualize Questions</h1>
            </div>
            <p>Now you don't have to explain, just send your photo</p>
            <p><b>Remaining Coins:</b> {st.session_state['coins']}</p>
        """, unsafe_allow_html=True)

    image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    user_prompt = st.text_input("Enter the prompt for image captioning:")

    if st.button("Visualize") and image and user_prompt:
        deduct_coin()  # Deduct 1 coin for image captioning
        load_image = Image.open(image)

        colLeft, colRight = st.columns(2)

        with colLeft:
            st.image(load_image.resize((800, 500)))

        caption_response = gemini_vision_response(model, user_prompt, load_image, knowledge_base)

        with colRight:
            st.info(caption_response)

elif user_picked == 'BPM Monitor':
    st.title("❤️ BPM Monitor Dashboard - Multi User")
    
    if not firebase_initialized:
        st.error("Firebase connection required for BPM monitoring")
        st.stop()
    
    # Get BPM data
    bmp_data = get_bmp_history()
    
    if not bmp_data:
        st.warning("No BPM data available. Make sure your ESP32 devices are connected and sending data.")
    else:
        # Get unique persons
        persons = list(set(item.get('person', 'unknown') for item in bmp_data))
        
        # Person selector
        if len(persons) > 1:
            selected_person = st.selectbox(
                "Select Person for Analysis:",
                options=['all'] + persons,
                format_func=lambda x: 'All Persons' if x == 'all' else x.title()
            )
        else:
            selected_person = persons[0] if persons else 'all'
        
        # Filter data based on selection
        if selected_person == 'all':
            display_data = bmp_data
        else:
            display_data = [item for item in bmp_data if item.get('person') == selected_person]
        
        if not display_data:
            st.warning(f"No data available for {selected_person}")
            st.stop()
        
        # Current BPM display
        latest_bmp = display_data[-1]['bpm'] if display_data else 0
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current BPM", f"{latest_bmp} bpm")
        
        with col2:
            avg_bpm = sum(item['bpm'] for item in display_data) / len(display_data)
            st.metric("Average BPM", f"{avg_bpm:.1f} bpm")
        
        with col3:
            st.metric("Total Records", len(display_data))
        
        # Multi-person summary
        if len(persons) > 1:
            st.subheader("📊 Multi-Person Summary")
            summary_cols = st.columns(len(persons))
            
            for i, person in enumerate(persons):
                person_data = [item for item in bmp_data if item.get('person') == person]
                if person_data:
                    person_avg = sum(item['bpm'] for item in person_data) / len(person_data)
                    with summary_cols[i]:
                        st.metric(
                            f"{person.title()}",
                            f"{person_avg:.1f} bpm",
                            delta=f"{len(person_data)} records"
                        )
        
        # BPM Chart
        st.subheader("📈 BPM Trend")
        chart = create_bmp_chart(bmp_data, selected_person)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Analysis
        st.subheader("🔍 BPM Analysis")
        analysis = analyze_bmp_data(bmp_data, selected_person)
        st.text_area("Analysis Report", analysis, height=400)
        
        # Data table
        # st.subheader("📊 Recent BPM Data")
        # if len(display_data) > 10:
        #     recent_data = display_data[-10:]
        # else:
        #     recent_data = display_data
        
        # df_display = pd.DataFrame([{
        #     'Person': item.get('person', 'unknown').title(),
        #     'Timestamp': item['timestamp'],
        #     'BPM': item['bpm'],
        #     'Status': 'Normal' if 60 <= item['bmp'] <= 100 else 'Abnormal'
        # } for item in reversed(recent_data)])
        
        # st.dataframe(df_display, use_container_width=True)
        
        # Export functionality
        if st.button("📄 Export BPM Data"):
            df_export = pd.DataFrame([{
                'person': item.get('person', 'unknown'),
                'timestamp': item['timestamp'],
                'datetime': item['datetime'],
                'bpm': item['bpm']
            } for item in bmp_data])
            
            csv = df_export.to_csv(index=False)
            filename_suffix = f"_{selected_person}" if selected_person != 'all' else "_all_persons"
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"bmp_data{filename_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Refresh data button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Auto-refresh every 30 seconds
        st.markdown("*Data automatically refreshes every minute*")
        
        # Health alerts
        if latest_bmp < 60 or latest_bmp > 100:
            if latest_bmp < 60:
                st.error(f"⚠️ Alert: Low heart rate detected ({latest_bmp} bpm) - Consider medical consultation")
            else:
                st.error(f"⚠️ Alert: High heart rate detected ({latest_bmp} bpm) - Consider medical consultation")
        else:
            st.success(f"✅ Current heart rate is normal ({latest_bmp} bpm)")

# Footer
st.markdown("---")
st.markdown("**HealthBot** - Created by Glenn & Bryan | Multi-User BPM Monitoring System")
st.markdown("*Connect your ESP32 devices to start monitoring heart rate data*")
