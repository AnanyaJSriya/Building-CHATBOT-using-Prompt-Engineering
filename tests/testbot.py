from src.curious_bot import get_response

def test_get_response():
    # Test if the bot correctly echoes back the input
    input_text = "Hello"
    response = get_response(input_text)
    assert "You said: Hello" in response
    assert "curious bot" in response
