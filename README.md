🤖 Curious: The Millennium Peer-Learning Bot
Curious is a high-performance voice bot designed for the Socratic Method of learning. Unlike standard AI assistants that provide answers, Curious acts as a peer student—listening to the user's explanation and asking deep, inquisitive questions to help the teacher (user) refine their own understanding.

🌟 Key Features
Socratic Intent Recognition: Built with custom system prompts to ensure the bot never gives answers, only asks high-level questions.

Low-Latency Voice UI: Features a custom Sand-Yellow & Teal Streamlit interface with near-instant speech-to-text and text-to-speech feedback.

Hybrid-Cloud Architecture: Scalable for consumer use via OpenAI's high-speed APIs, while maintaining architectural support for local-only deployment.

Market-Ready Reliability: Overcomes common AI limitations including ambiguity, uncertainty, and "hallucination loops."

🛠️ Technical Stack
Language: Python 3.11

LLM Engine: GPT-4o-mini / Llama-3 (Local-capable)

Framework: Streamlit

DevOps: GitHub Actions (CI/CD), Pytest

Voice: Whisper-1 (STT), OpenAI TTS-1

🚀 Quick Start
Clone the Repo:

Bash
git clone https://github.com/AnanyaJSriya/Building-CHATBOT-using-Prompt-Engineering
cd Building-CHATBOT-using-Prompt-Engineering
Install Dependencies:

Bash
pip install -r requirements.txt

Set Secrets: Ensure your OPENAI_API_KEY is set in your environment or Streamlit Secrets.

Run the App:

Bash
streamlit run src/curious_bot.py

🧪 Quality Assurance
This project includes an automated test suite. To verify the bot's logic:

Bash
pytest tests/

Developed as a Millennium Project to push the boundaries of Peer-to-Peer AI Learning.
