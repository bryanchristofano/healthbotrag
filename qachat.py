import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from pathlib import Path
from PIL import Image
from streamlit_option_menu import option_menu
#from st_paywall import add_auth

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

# Generate response using the Gemini model and knowledge base
def generate_response(input_text, knowledge_base):
    knowledge_summary = " ".join(knowledge_base)
    full_prompt = f"{knowledge_summary}\nUser: {input_text}\nAssistant:"
    if input_text.lower() in ["siapa namamu", "who are you", "siapa kamu", "kamu siapa", "siapa anda", "siapa kamu?", "siapa namamu?", "who are you?", "kamu siapa?", "siapa anda?", "kamu adalah apa", "kamu adalah apa?"]:
        return "Saya adalah HealthBot buatan Glenn dan Bryan berdasarkan knowledge base yang diberikan oleh mereka. Terimakasih Glenn dan Bryan."
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

# Load knowledge base from PDFs
knowledge_base = load_pdfs_from_folder("docs")

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

# Sidebar menu for choosing between Manual Guide, ChatBot, and Image Captioning
with st.sidebar:
    user_picked = option_menu(
        "HealthBot",
        ["Manual Guide", "Chat Doctor", "Image Solutions"],
        menu_icon="robot",
        icons=["info-circle", "chat-dots-fill", "image-fill"],
        default_index=0
    )

# Doctor profile picture
doctorPic = Image.open(doctorpic)
doctorPic = doctorPic.resize((200, 200))


# Display remaining coins
st.sidebar.write(f"Remaining Coins: {st.session_state['coins']}")

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
        5. When your coins run out, you will be redirected to the subscription page.
    """)

    st.header("How to Use Image Solutions")
    st.write("""
        1. Go to the 'Image Solutions' section from the sidebar.
        2. Upload an image related to your health query.
        3. Enter a prompt describing what you want to know about the image.
        4. The system will generate a response using the Gemini Vision model.
        5. Each image submission deducts 1 coin.
    """)

elif user_picked == 'Chat Doctor':
    model = gemini_pro()

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
        st.markdown(f"""
            <div class="doctor-header">
                <h1>Dr. Healthbot</h1>
            </div>
            <p>I specialize in skin and genital health</p>
            <p><b>Remaining Coins:</b> {st.session_state['coins']}</p>
        """, unsafe_allow_html=True)

    # Display the chat history
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Something...")
    if user_prompt:
        deduct_coin()  # Deduct 1 coin for every prompt
        st.session_state.chat_history.append(("user", user_prompt))
        st.chat_message("user").markdown(user_prompt)

        gemini_response = generate_response(user_prompt, knowledge_base, st.session_state.chat_history)


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
