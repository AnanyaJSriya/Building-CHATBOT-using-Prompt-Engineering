import os
import io
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# Load environment variables safely
load_dotenv()

# Setup OpenAI client with a fallback for testing environments
api_key = os.getenv("OPENAI_API_KEY", "missing_key")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are 'Curious', a peer-learning assistant. 
Your goal is to be taught by the user. 
Ask intuitive questions that force the user to explain 'why'.
Be concise and stay in character as a curious learner.
"""

def transcribe_audio(audio_bytes):
    if not audio_bytes:
        return ""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcript.text

def generate_voice(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    return response.content

def get_chat_response(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="Curious Bot", page_icon="🎙️")
    st.title("🎙️ Teach Curious")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Voice Input Section
    audio_input = mic_recorder(
        start_prompt="Click to Speak 🎤",
        stop_prompt="Stop & Send 📤",
        key='recorder'
    )

    if audio_input and 'bytes' in audio_input:
        with st.spinner("Curious is listening..."):
            user_text = transcribe_audio(audio_input['bytes'])
            st.session_state.messages.append({"role": "user", "content": user_text})

            bot_text = get_chat_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": bot_text})

            audio_response = generate_voice(bot_text)
            st.audio(audio_response, format="audio/mp3", autoplay=True)

    # Display Conversation History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if __name__ == "__main__":
    main()
