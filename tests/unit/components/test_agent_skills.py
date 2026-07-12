import pytest

from daystrom import Provider
from daystrom.components import LLM, Agent, Context, LLMResponse, Tool
from daystrom.permissions import ReadPermission


class ConcreteLLM(LLM):
    def invoke(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(text="", tool_calls=[])

    def track_usage(self, *args, **kwargs):
        pass


@pytest.fixture
def llm():
    return ConcreteLLM(provider=Provider.OPENROUTER, model="test-model")


def write_skill(root, name: str, body: str = "Use the skill."):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use for agent tests.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_agent_disables_skills_with_empty_allowlist(tmp_path, monkeypatch, llm):
    write_skill(tmp_path / ".daystrom" / "skills", "sample-skill")
    monkeypatch.chdir(tmp_path)

    agent = Agent(llm=llm, skills=[])

    assert agent.skills == {}
    assert "read_file" in agent.tools
    assert "skill" not in agent.tools
    # print(agent.tools)
    assert agent.context.messages == []


def test_agent_injects_catalog_registers_skill_tools(tmp_path, monkeypatch, llm):
    write_skill(tmp_path / ".daystrom" / "skills", "sample-skill")
    monkeypatch.chdir(tmp_path)

    agent = Agent(llm=llm)

    assert "sample-skill" in agent.skills
    assert "read_file" in agent.tools
    assert "skill" in agent.tools
    assert len(agent.context.messages) == 1
    assert agent.context.messages[0].role == "system"
    assert "<available_skills>" in agent.context.messages[0].text
    assert "sample-skill" in agent.context.messages[0].text


def test_agent_allowlists_skills_by_name(tmp_path, monkeypatch, llm):
    write_skill(tmp_path / ".daystrom" / "skills", "sample-skill")
    write_skill(tmp_path / ".daystrom" / "skills", "other-skill")
    monkeypatch.chdir(tmp_path)

    agent = Agent(llm=llm, skills=["sample-skill"])

    assert list(agent.skills) == ["sample-skill"]
    assert "other-skill" not in agent.context.messages[0].text


def test_agent_skill_tool_returns_content(tmp_path, monkeypatch, llm):
    write_skill(tmp_path / ".daystrom" / "skills", "sample-skill", body="Do the thing.")
    monkeypatch.chdir(tmp_path)
    agent = Agent(llm=llm)

    assert "skill" in agent.toolPermissions
    content = agent.tools["skill"].call(
        "sample-skill", permissions=agent.toolPermissions["skill"], skills=agent.skills
    )

    assert '<skill-content name="sample-skill">' in content
    assert "Do the thing." in content


# TODO: Write a test for preventing duplicate skill activations
#
# def test_agent_activate_skill_deduplicates(tmp_path, monkeypatch, llm):
#    write_skill(tmp_path / ".daystrom" / "skills", "sample-skill")
#    monkeypatch.chdir(tmp_path)
#    agent = Agent(llm=llm)
#
#    agent.activate_skill("sample-skill")
#    message_count = len(agent.context.messages)
#    second = agent.activate_skill("sample-skill")
#
#    assert second == "Skill already activated: sample-skill"
#    assert len(agent.context.messages) == message_count

# TODO: Delete pending review
#
# def test_tool_activate_skill_does_not_add_system_message(tmp_path, monkeypatch, llm):
#    write_skill(tmp_path / ".daystrom" / "skills", "sample-skill")
#    monkeypatch.chdir(tmp_path)
#    agent = Agent(llm=llm)
#    message_count = len(agent.context.messages)
#
#    content = agent.tools["activate_skill"].call("sample-skill")
#
#    assert '<skill_content name="sample-skill">' in content
#    assert len(agent.context.messages) == message_count

# TODO: Delete pending review. I think this is redundant with test_permissions since it effectively would have to end up doing the same thing; calling the tool directly with a passed Permission. Not sure how to test that without going through an invoke loop, which wouldn't be a unit test anymore. unless we can mock around it somehow?
#
# def test_agent_read_file_uses_permission_policy(tmp_path, monkeypatch, llm):
#    allowed_file = tmp_path / "allowed.txt"
#    allowed_file.write_text("one\ntwo\nthree\n", encoding="utf-8")
#    workdir = tmp_path / "workdir"
#    workdir.mkdir()
#    monkeypatch.chdir(workdir)
#    policy = ReadPermissionPolicy([tmp_path])
#    agent = Agent(llm=llm, permission_policy=policy, skills=[])
#
#    result = agent.tools["read_file"].call(str(allowed_file), offset=2, limit=1)
#
#    assert "2: two" in result
#    assert "1: one" not in result
#    assert "read_file" in agent.llm.tools


def test_agent_preserves_tools_dict_identity(tmp_path, monkeypatch, llm):
    monkeypatch.chdir(tmp_path)
    tools = {
        "custom": Tool(
            callable=lambda: "ok",
            name="custom",
            description="Custom tool",
            params={},
        )
    }

    agent = Agent(llm=llm, tools=tools, skills=[])

    assert agent.tools is tools
    assert agent.llm.tools is tools
    assert "custom" in tools
