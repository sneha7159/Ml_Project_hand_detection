import speech_recognition as sr
from gtts import gTTS
import io
import threading
import streamlit as st

class VoiceAssistant:
    def __init__(self, name="Nova"):
        self.name = name
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        
        # Adjust for ambient noise
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
        except:
            pass  # Handle cases where microphone is not available
    
    def speak(self, text):
        """Convert text to speech using Streamlit audio player"""
        try:
            tts = gTTS(text=text, lang='en')
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            # Display audio player in Streamlit
            st.audio(audio_bytes, format='audio/mp3')
            
        except Exception as e:
            st.write(f"{self.name}: {text}")  # Fallback to text
    
    def listen(self):
        """Listen for voice commands"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            text = self.recognizer.recognize_google(audio)
            return text.lower()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
        except Exception as e:
            return ""
    
    def process_command(self, command):
        """Process voice commands"""
        if "hello" in command or "hi" in command:
            self.speak(f"Hello! I'm {self.name}, your learning assistant.")
        elif "help" in command:
            self.speak("I can help you navigate through games, explain concepts, or celebrate your achievements!")
        elif "what can you do" in command:
            self.speak("I can assist you with all the educational games on Shikshakhel. Just ask for help in any game!")
        elif "thank you" in command:
            self.speak("You're welcome! Keep up the great learning!")
        else:
            self.speak("I'm not sure how to help with that. Try asking for help in a specific game.")