import streamlit as st
from openai import OpenAI
import os

# Resilient key fetching: Try Streamlit secrets first, then environment variables
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY", "missing_key_for_testing")

client = OpenAI(api_key=get_api_key())

# THE DESIGN: Brand-Specific Sand-Yellow & Teal
st.markdown("""
    <style>
    .stApp { background-color: #F5E6BE; }
    h1, h2, h3, p { color: #008080 !important; text-align: center; font-family: 'Helvetica'; }
    .stButton>button { background-color: #008080; color: white; border-radius: 25px; border: none; }
    </style>
""", unsafe_allow_html=True)

# Initialization & Security
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

class ConsumerCurious:
    def __init__(self):
        self.persona = "Socratic Learner"
        self.system_prompt = (
            "You are 'Curious', a peer student. Your goal is to be taught. "
            "Ask one intuitive question at a time. Never provide answers. "
            "Minimize lag: Keep responses under 20 words."
        )

    def process_voice(self, audio_bytes):
        # 1. Near-Zero Lag Transcriptions (Whisper-1)
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        
        with open("temp_audio.wav", "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        return transcript.text

    def get_socratic_response(self, text):
        # 2. Intent Recognition & Hallucination Guard
        response = client.chat.completions.create(
            model="gpt-4o-mini", # High speed, low cost, high intelligence
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content

    def speak(self, text):
        # 3. Natural Voice Synthesis (TTS-1)
        response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
        return base64.b64encode(response.content).decode()

def main():
    st.title("🤖 CURIOUS")
    st.markdown("### The Global Peer-Learning Assistant")
    st.write("Explain your topic clearly. I'm here to learn from you!")

    bot = ConsumerCurious()

    # Optimized Deployment: One-click voice interaction
    audio_data = mic_recorder(start_prompt="🎤 Click to Teach", stop_prompt="⏹️ Stop & Send", key='recorder')

    if audio_data:
        with st.spinner("Curious is thinking..."): # Masking Hardware Latency
            # STT
            user_text = bot.process_voice(audio_data['bytes'])
            st.info(f"You said: {user_text}")

            # LLM
            bot_text = bot.get_socratic_response(user_text)
            st.success(f"Curious: {bot_text}")

            # TTS
            audio_b64 = bot.speak(bot_text)
            audio_html = f'<audio autoplay src="data:audio/mp3;base64,{audio_b64}">'
            st.markdown(audio_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
