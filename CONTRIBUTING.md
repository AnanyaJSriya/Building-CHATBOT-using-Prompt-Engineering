# Contributing to Curious

Thank you for your interest in contributing! Here's how you can help:

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Keep functions focused and small

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Commit Messages

Use semantic commit messages:
- `feat: Add new feature`
- `fix: Fix bug`
- `docs: Update documentation`
- `test: Add tests`
- `refactor: Refactor code`
```

**4. requirements.txt** (properly formatted)
```
openai>=1.0.0
python-dotenv>=1.0.0
speech_recognition>=3.10.0
pyttsx3>=2.90
sounddevice>=0.4.6
numpy>=1.24.0
```

**5. requirements-dev.txt**
```
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
pylint>=2.17.0
mypy>=1.0.0
pre-commit>=3.3.0
