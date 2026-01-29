"""Tests for the CuriousBot class."""

import pytest
from src.curious_bot import CuriousBot

@pytest.fixture
def bot():
    """Create a CuriousBot instance for testing."""
    return CuriousBot(api_key="test_key")

def test_bot_initialization(bot):
    """Test that bot initializes correctly."""
    assert bot is not None
    assert bot.api_key == "test_key"
    assert len(bot.conversation_history) >= 0

def test_process_input(bot):
    """Test input processing generates valid response."""
    response = bot.process_input("Photosynthesis is how plants make food")
    assert isinstance(response, str)
    assert len(response) > 0

def test_conversation_context(bot):
    """Test that conversation history is maintained."""
    bot.process_input("First message")
    bot.process_input("Second message")
    assert len(bot.conversation_history) > 0

# Add more tests...
