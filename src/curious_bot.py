import os
import io
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are 'Curious', a peer-learning assistant. Your goal is NOT to teach, but to be taught.
1. The user will explain a topic via voice. 
2. Act as a curious learner. Ask ONE deep-diving question that forces the user to explain 'why'.
3. If they are wrong, gently challenge them. Do not hallucinate.
4. Keep voice responses very short (1-2 sentences) to keep the conversation fluid.
"""

def transcribe_audio(audio_bytes):
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
        voice="nova", # Professional, friendly voice
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
    st.set_page_config(page_title="Curious Voice Bot", page_icon="🎙️")
    st.title("🎙️ Teach Curious (Voice Mode)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # UI for Voice Input
    st.write("Click the mic and start explaining your topic:")
    audio_input = mic_recorder(start_prompt="Start Recording 🎤", stop_prompt="Stop & Send 📤", key='recorder')

    if audio_input:
        user_text = transcribe_audio(audio_input['bytes'])
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Get LLM Response
        bot_text = get_chat_response(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": bot_text})

        # Generate Voice Response
        audio_response = generate_voice(bot_text)
        st.audio(audio_response, format="audio/mp3", autoplay=True)

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if __name__ == "__main__":
    main()
