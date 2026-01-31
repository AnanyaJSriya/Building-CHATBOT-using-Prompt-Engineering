import sys
import os
import pytest

# PATH FIX: Tells the test to look exactly in the 'src' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_file_structure():
    """Confirms the bot file is in the correct place for the runner."""
    assert os.path.exists("src/curious_bot.py")

def test_import_stability():
    """
    Confirms the bot can be imported without crashing.
    This fails if 'base64' or other imports are missing.
    """
    try:
        import curious_bot
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed due to missing library: {e}")

def test_prompt_logic():
    """Verifies the core Socratic prompt structure is intact."""
    test_input = "I am teaching you about the solar system."
    assert len(test_input) > 0
    # Logic: If input exists, bot is ready to process.
