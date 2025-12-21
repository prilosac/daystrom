import pytest

from daystrom.components.openrouter import OpenRouterChat


@pytest.fixture
def client():
    return OpenRouterChat(model="anthropic/claude-haiku-4.5")


@pytest.fixture
def message():
    return "Give me a short response as a test that API functionality is working"


def test_invoke_stream(client, message):
    res = "".join(client.invoke_stream(message))
    assert isinstance(res, str)
    assert res != ""


def test_invoke(client, message):
    res = client.invoke(message)
    assert isinstance(res, str)
    assert res != ""


@pytest.mark.asyncio
async def test_ainvoke_stream(client, message):
    res = "".join([chunk async for chunk in client.ainvoke_stream(message)])
    assert isinstance(res, str)
    assert res != ""


@pytest.mark.asyncio
async def test_ainvoke(client, message):
    res = await client.ainvoke(message)
    assert isinstance(res, str)
    assert res != ""
