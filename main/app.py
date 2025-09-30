import streamlit as st
import sqlite3
import json
import base64
from voice_assistant import VoiceAssistant
import games.alphabet_snake as alphabet_snake
import games.color_target as color_target
import games.voice_challenge as voice_challenge
import games.emotion_learning as emotion_learning
import games.math_challenge as math_challenge
import games.geometry_gesture as geometry_gesture
import games.science_lab as science_lab
import games.word_builder as word_builder
import games.drawing_recognition as drawing_recognition
import games.music_rhythm as music_rhythm
import games.puzzle_solver as puzzle_solver
from database import init_db, get_user_data, save_game_progress

# Initialize database
init_db()

# Initialize voice assistant
assistant = VoiceAssistant()

# Page configuration
st.set_page_config(
    page_title="Shikshakhel - AI Learning Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open("static/css/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        # Fallback CSS if file not found
        st.markdown("""
        <style>
        body {
            background-color: #0a0a2a;
            color: #ffffff;
        }
        .stButton>button {
            background: linear-gradient(45deg, #2196f3, #1976d2);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

load_css()

# Main application
def main():
    # Sidebar navigation
    st.sidebar.title("🚀 Shikshakhel")
    st.sidebar.markdown("### AI Learning Platform")
    
    # User authentication
    if 'user_id' not in st.session_state:
        user_name = st.sidebar.text_input("Enter your name to begin:")
        if st.sidebar.button("Start Learning"):
            if user_name:
                st.session_state.user_id = user_name
                st.session_state.page = "dashboard"
                assistant.speak(f"Welcome to Shikshakhel, {user_name}! I'm Nova, your learning assistant. Let's explore the amazing world of knowledge together!")
            else:
                st.sidebar.error("Please enter your name.")
    
    if 'user_id' in st.session_state:
        user_id = st.session_state.user_id
        menu = ["Learning Dashboard", "Alphabet Snake", "Color Target", "Voice Challenge", 
                "Emotion Learning", "Math Challenge", "Geometry Gesture", "Science Lab",
                "Word Builder", "Drawing Recognition", "Music Rhythm", "Puzzle Solver"]
        
        choice = st.sidebar.selectbox("Navigate to:", menu)
        
        # Voice assistant control
        if st.sidebar.button("🔊 Ask Nova"):
            assistant.speak("How can I help you with your learning journey today?")
        
        # Display current page
        if choice == "Learning Dashboard":
            show_dashboard(user_id)
        elif choice == "Alphabet Snake":
            alphabet_snake.game(user_id, assistant)
        elif choice == "Color Target":
            color_target.game(user_id, assistant)
        elif choice == "Voice Challenge":
            voice_challenge.game(user_id, assistant)
        elif choice == "Emotion Learning":
            emotion_learning.game(user_id, assistant)
        elif choice == "Math Challenge":
            math_challenge.game(user_id, assistant)
        elif choice == "Geometry Gesture":
            geometry_gesture.game(user_id, assistant)
        elif choice == "Science Lab":
            science_lab.game(user_id, assistant)
        elif choice == "Word Builder":
            word_builder.game(user_id, assistant)
        elif choice == "Drawing Recognition":
            drawing_recognition.game(user_id, assistant)
        elif choice == "Music Rhythm":
            music_rhythm.game(user_id, assistant)
        elif choice == "Puzzle Solver":
            puzzle_solver.game(user_id, assistant)
    else:
        show_landing_page()

def show_landing_page():
    st.markdown("""
    <div style="text-align: center; padding: 100px 20px;">
        <h1>Welcome to Shikshakhel</h1>
        <h2>AI-Powered Learning Adventure</h2>
        <p>Embark on an educational journey through space with Nova, your AI assistant!</p>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard(user_id):
    st.title(f"Learning Dashboard - Welcome {user_id}!")
    
    # Get user progress
    progress = get_user_data(user_id)
    
    # Display progress
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Games Played", progress.get('games_played', 0))
    with col2:
        st.metric("Total Score", progress.get('total_score', 0))
    with col3:
        st.metric("Learning Level", progress.get('level', 'Beginner'))
    
    # Game cards
    st.subheader("Educational Games")
    cols = st.columns(3)
    games = [
        ("Alphabet Snake", "🐍", "Catch alphabets to form words"),
        ("Color Target", "🎯", "Improve reaction time and color recognition"),
        ("Voice Challenge", "🎤", "Practice voice commands and pronunciation"),
        ("Emotion Learning", "😊", "Recognize and understand emotions"),
        ("Math Challenge", "➗", "Solve fun math problems"),
        ("Geometry Gesture", "🔺", "Create shapes with gestures"),
        ("Science Lab", "🔬", "Perform virtual experiments"),
        ("Word Builder", "🔠", "Form words with gestures and voice"),
        ("Drawing Recognition", "✏️", "AI recognizes drawn objects"),
        ("Music Rhythm", "🎵", "Clap to rhythms and patterns"),
        ("Puzzle Solver", "🧩", "Solve various puzzle types")
    ]
    
    for i, (name, icon, desc) in enumerate(games):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background: rgba(25, 25, 55, 0.8); border: 1px solid #448aff; border-radius: 10px; padding: 20px; margin: 10px 0;">
                <h3>{icon} {name}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Play {name}", key=f"btn_{name}"):
                st.session_state.page = name
                st.experimental_rerun()

if __name__ == "__main__":
    main()