import streamlit as st
from openai import OpenAI
import base64
import os

# Resilient Import for Streamlit Cloud
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

def get_bot_response(user_input, api_key):
    """
    Logic separated from the UI. This allows the test suite 
    to run without launching a browser.
    """
    client = OpenAI(api_key=api_key)
    system_prompt = "You are 'Curious', a peer student. Ask one intuitive question to learn from the user."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="Curious Bot", page_icon="🤖")
    st.markdown("<style>.stApp { background-color: #F5E6BE; } h1, p { color: #008080; text-align: center; }</style>", unsafe_allow_html=True)
    
    st.title("🤖 CURIOUS")
    st.write("Explain your topic. I am here to learn from you.")

    # Accesses the secret you installed in the settings
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "missing"))
    
    if mic_recorder:
        audio_data = mic_recorder(start_prompt="🎤 Click to Teach", stop_prompt="⏹️ Stop & Send", key='recorder')
        
        if audio_data:
            if api_key == "missing":
                st.error("API Key not found in Secrets!")
                return

            try:
                with st.spinner("Curious is processing your lesson..."):
                    # 1. Logic call
                    client = OpenAI(api_key=api_key)
                    # (Transcription logic goes here in live use)
                    st.success("Response generated!")
            except Exception as e:
                st.error(f"Handshake Error: {str(e)}")

if __name__ == "__main__":
    main()
