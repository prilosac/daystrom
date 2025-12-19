# Agents Guide

## Build & Test Commands
- Install: `uv sync && uv pip install -e .`
- Run all tests: `uv run pytest`
- Run single test file: `uv run pytest tests/integration/nodes/openrouter/test_chat.py`
- Run single test: `uv run pytest tests/integration/nodes/openrouter/test_chat.py::test_invoke`
- Type check: `uv run pyright`

## Code Style
- **Imports**: stdlib first, third-party second, local last. Use absolute imports (`from daystrom.context import Context`).
- **Types**: Use type hints on parameters and returns. Use modern generics (`list[T]` not `List[T]`).
- **Naming**: PascalCase for classes, snake_case for functions/methods, SCREAMING_SNAKE_CASE for constants. Prefix async methods with `a` (e.g., `ainvoke`).
- **Async**: Use `async`/`await` patterns. Async generators for streaming (`async for`, `yield`).
- **Testing**: Use pytest with `@pytest.mark.asyncio` for async tests. Test files/functions prefixed with `test_`.

## Project Structure
- Source code: `src/daystrom/`
- Tests: `tests/`
- Python 3.12, managed with `uv`

## Architecture Notes
- **LLM Nodes**: Chat nodes (e.g., `OpenRouterChat`) should expose their underlying client for composition with other tools like Instructor.
- **Structured Output**: Use Pydantic models for schemas. For dynamic/user-defined schemas, use `pydantic.create_model()`. Wrap LLM clients with Instructor for structured output with retries.
- **Composability**: Prefer designs where nodes can be passed into other nodes (e.g., `StructuredOutput(chat=my_chat_node, response_model=MySchema)`).
