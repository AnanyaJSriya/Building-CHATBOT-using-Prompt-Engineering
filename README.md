# Curious - AI Voice Learning Bot 🎓🗣️

An intelligent voice-based chatbot that acts as a curious student, helping users practice teaching and learning through interactive conversations.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🌟 Overview

Curious is a voice-enabled AI chatbot designed to help users:
- Practice teaching skills by explaining concepts to an AI "student"
- Learn topics more deeply through the Feynman Technique
- Engage in natural, voice-based conversations
- Receive curious questions that promote deeper understanding

## ✨ Features

- 🎤 **Voice Interaction**: Full voice input and output capabilities
- 🤔 **Curious Student Persona**: AI asks thoughtful follow-up questions
- 📚 **Learning-Focused**: Designed around proven teaching methodologies
- 💬 **Natural Conversations**: Context-aware dialogue management
- 🔧 **Prompt Engineering**: Built using advanced prompt engineering techniques

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Microphone and speakers
- OpenAI API key (or compatible API)

### Installation

1. Clone the repository
```bash
git clone https://github.com/AnanyaJSriya/Building-CHATBOT-using-Prompt-Engineering.git
cd Building-CHATBOT-using-Prompt-Engineering
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up your API key
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Usage

Run the main chatbot:
```bash
python CuriousVoice.py
```

**Example Interaction:**
```
You: "I'd like to explain photosynthesis"
Curious: "Great! I'm excited to learn about photosynthesis. 
         Can you start by telling me what it is in simple terms?"
You: [Your explanation]
Curious: "Interesting! So if plants need sunlight, what happens at night?"
```

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [Prompt Engineering Guide](docs/prompts.md)
- [Voice Integration](docs/voice.md)
- [API Reference](docs/api.md)

## 🏗️ Project Structure
```
Building-CHATBOT-using-Prompt-Engineering/
├── src/                    # Source code
│   ├── curious_bot.py     # Main bot logic
│   ├── prompts/           # Prompt templates
│   ├── voice/             # Voice processing
│   └── utils/             # Utilities
├── tests/                 # Test files
├── examples/              # Usage examples
├── docs/                  # Documentation
├── .env.example          # Environment template
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🧪 Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Linting
flake8 src/

# Formatting
black src/
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📝 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👤 Author

**Ananya J Sriya**
- GitHub: [@AnanyaJSriya](https://github.com/AnanyaJSriya)

## 🙏 Acknowledgments

- Built using prompt engineering principles
- Inspired by the Feynman Technique for learning
- Voice processing powered by [specify library]

## 📊 Project Status

**Current Version**: 2.0.0  
**Status**: Active Development

---

*Found this helpful? Give it a ⭐️!*
```

#### **B. Reorganize File Structure**

**Action Items:**
1. Delete redundant attempt files or move to `archive/` folder
2. Rename files consistently (use snake_case)
3. Create proper directory structure:
```
Building-CHATBOT-using-Prompt-Engineering/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── curious_bot.py          # Consolidated main file
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system_prompts.py
│   │   └── templates.py
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── speech_to_text.py
│   │   └── text_to_speech.py
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_bot.py
│   ├── test_prompts.py
│   └── test_voice.py
├── examples/
│   ├── basic_usage.py
│   └── custom_prompts.py
├── docs/
│   ├── architecture.md
│   ├── prompts.md
│   └── api.md
├── archive/                     # Old attempt files
│   ├── curiousattempt1.py
│   ├── curious_version1.py
│   └── Curiousversion2.py
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── pyproject.toml
```

#### **C. Add Essential Files**

**1. .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Audio/Voice
*.wav
*.mp3
temp_audio/

# Logs
*.log
logs/

# Testing
.coverage
.pytest_cache/
htmlcov/
```

**2. LICENSE (MIT)**
```
MIT License

Copyright (c) 2026 Ananya J Sriya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
