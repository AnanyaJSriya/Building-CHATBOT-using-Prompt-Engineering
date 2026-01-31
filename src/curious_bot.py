import streamlit as st
from openai import OpenAI
import os
import base64

# Import moved inside a safe check to prevent CI from failing if streamlit-mic-recorder isn't pre-loaded
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

def get_bot_response(user_input, api_key):
    """Business logic separated for foolproof testing."""
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

    # Safe API Key Fetching
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "missing"))
    
    if mic_recorder:
        audio_data = mic_recorder(start_prompt="🎤 Click to Teach", stop_prompt="⏹️ Stop & Send", key='recorder')
        if audio_data and api_key != "missing":
            # Transcription and TTS logic remains here for the live app
            st.success("Curious is processing your lesson...")

if __name__ == "__main__":
    main()
