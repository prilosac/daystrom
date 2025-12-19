import pytest

from daystrom.nodes.openrouter.chat import OpenRouterChat

OPENROUTER_KEY = ""


@pytest.fixture
def message():
    return "Give me a short response as a test that API functionality is working"


def test_flow(message):
    client = OpenRouterChat(api_key=OPENROUTER_KEY, model="anthropic/claude-haiku-4.5")
    res = client.invoke(message)
    print(res)
    client2 = OpenRouterChat(api_key=OPENROUTER_KEY, model="anthropic/claude-haiku-4.5")
    res2 = client2.invoke(res)
    print(res2)
    assert isinstance(res, str) and isinstance(res2, str)
