from dotenv import load_dotenv
import streamlit as st


# Load environment variables from .env file
load_dotenv()


def get_response(user_input):
    """
    Placeholder function for chatbot logic.
    """
    return f"You said: {user_input}. I am a curious bot learning prompt engineering!"


def main():
    st.set_page_config(page_title="Curious Chatbot", page_icon="🤖")
    st.title("Building a Chatbot using Prompt Engineering")
    st.write("Welcome! Type something below to chat with the bot.")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response
        response = get_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
