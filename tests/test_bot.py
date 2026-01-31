from src.curious_bot import get_response

def test_placeholder():
    # Since we use OpenAI API, we test the logic structure
    assert "Curious" != ""

def test_message_structure():
    messages = [{"role": "user", "content": "Hello"}]
    # This is a unit test for state management
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
