# Contributing to FastRAG-Vector-Search

We welcome all contributions! Please follow these guidelines:

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dependencies: `pip install -r requirements.txt`
4. Make your changes
5. Run tests: `pytest tests/ -v`
6. Commit: `git commit -m "feat: add my feature"`
7. Push: `git push origin feature/my-feature`
8. Open a Pull Request

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Add type hints to all functions
- Include docstrings for all public classes and methods
- Keep lines ≤ 100 characters

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation update |
| `perf:` | Performance improvement |
| `test:` | Adding or fixing tests |
| `refactor:` | Code refactoring |
