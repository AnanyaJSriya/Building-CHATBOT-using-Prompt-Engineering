import streamlit as st
from openai import OpenAI
import base64
import os

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

def main():
    st.set_page_config(page_title="Curious Bot", layout="centered")
    st.markdown("<style>.stApp { background-color: #F5E6BE; } h1, p { color: #008080; text-align: center; }</style>", unsafe_allow_html=True)
    
    st.title("🤖 CURIOUS")
    st.write("Explain your topic. I am here to learn from you.")

    # Secure Secret Check
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Please add OPENAI_API_KEY to Streamlit Secrets.")
        return
    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if mic_recorder:
        # The 'key' ensures the widget state is preserved
        audio_data = mic_recorder(start_prompt="🎤 Click to Teach", stop_prompt="⏹️ Stop & Send", key='recorder')
        
        if audio_data and len(audio_data['bytes']) > 0:
            try:
                with st.spinner("Curious is listening..."):
                    # 1. Save and Transcribe
                    with open("temp_audio.wav", "wb") as f:
                        f.write(audio_data['bytes'])
                    
                    with open("temp_audio.wav", "rb") as f:
                        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
                    
                    # VALIDATION: Show what the bot heard
                    st.info(f"You said: {transcript.text}")

                    # 2. Socratic Response
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a student. Ask one short curious question to the teacher. Do not answer them."},
                            {"role": "user", "content": transcript.text}
                        ]
                    )
                    reply = response.choices[0].message.content
                    st.success(f"Curious: {reply}")

                    # 3. Voice Output with Fallback
                    speech = client.audio.speech.create(model="tts-1", voice="nova", input=reply)
                    b64 = base64.b64encode(speech.content).decode()
                    
                    # Display Audio Player (Fallback if autoplay fails)
                    st.audio(speech.content, format="audio/mp3")
                    
                    # Autoplay script
                    md = f'<audio autoplay src="data:audio/mp3;base64,{b64}">'
                    st.markdown(md, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        elif audio_data:
             st.warning("No audio detected. Please check your mic permissions.")
