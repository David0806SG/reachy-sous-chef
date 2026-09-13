import base64

from conftest import ScriptedLLM, text_response, tool_response

from sous_chef.brain.agent import SousChefAgent
from sous_chef.brain.prompts import build_system_prompt
from sous_chef.brain.tools import TOOLS, ToolContext, ToolDispatcher
from sous_chef.kitchen.recipes import RecipeBook
from sous_chef.kitchen.timers import TimerManager


def make_agent(robot, settings, responses, sleep_flag=None):
    ctx = ToolContext(
        robot=robot,
        timers=TimerManager(lambda t: None),
        recipes=RecipeBook(settings.recipes_dir),
        on_sleep=lambda: sleep_flag.append(True) if sleep_flag is not None else None,
    )
    llm = ScriptedLLM(responses)
    agent = SousChefAgent(llm, build_system_prompt("David"), ToolDispatcher(ctx), history_turns=6)
    return agent, llm, ctx


def test_plain_text_turn(robot, settings):
    agent, llm, _ = make_agent(robot, settings, [text_response("Three minutes a side, chef.")])
    turn = agent.respond("how long for the sear", language="en")
    assert turn.spoken == "Three minutes a side, chef."
    assert turn.tool_calls == []
    assert llm.calls[0]["messages"][0]["content"].endswith("(spoken in English)")
    assert llm.calls[0]["tools"] is TOOLS
    assert "David" in llm.calls[0]["system"]


def test_tool_round_trip_timer(robot, settings):
    agent, llm, ctx = make_agent(
        robot,
        settings,
        [
            tool_response("set_timer", {"label": "pasta", "seconds": 540}, text="Setting it."),
            text_response("Nine minutes for the pasta, chef."),
        ],
    )
    turn = agent.respond("Reachy set a pasta timer for nine minutes", language="en")
    assert turn.tool_calls == ["set_timer"]
    assert turn.spoken == "Setting it. Nine minutes for the pasta, chef."
    assert [t.label for t in ctx.timers.active()] == ["pasta"]
    # the tool_result went back to the model with the right id
    result_msg = llm.calls[1]["messages"][-1]
    assert result_msg["role"] == "user"
    assert result_msg["content"][0]["tool_use_id"] == "tu_1"
    assert "9 minutes" in result_msg["content"][0]["content"]
    ctx.timers.cancel_all()


def test_camera_tool_returns_image_block(settings, robot):
    robot.jpeg = b"\xff\xd8fakejpeg"
    agent, llm, _ = make_agent(
        robot,
        settings,
        [
            tool_response("look", {"where": "counter"}, tool_id="tu_a"),
            tool_response("take_a_look", {"question": "is the sear done"}, tool_id="tu_b"),
            text_response("That crust is there, flip it."),
        ],
    )
    turn = agent.respond("look at this", language="en")
    assert turn.tool_calls == ["look", "take_a_look"]
    assert "look:counter" in robot.events and "capture" in robot.events
    image_result = llm.calls[2]["messages"][-1]["content"][0]["content"]
    assert image_result[1]["type"] == "image"
    assert base64.b64decode(image_result[1]["source"]["data"]) == b"\xff\xd8fakejpeg"


def test_camera_unavailable(settings, robot):
    robot.jpeg = None
    agent, llm, _ = make_agent(
        robot,
        settings,
        [tool_response("take_a_look", {"question": "x"}), text_response("Can't see right now.")],
    )
    agent.respond("look", language="en")
    assert "no frame" in llm.calls[1]["messages"][-1]["content"][0]["content"]


def test_recipe_tools(settings, robot):
    agent, llm, _ = make_agent(
        robot,
        settings,
        [
            tool_response("list_recipes", {}, tool_id="a"),
            tool_response("load_recipe", {"name": "ribeye"}, tool_id="b"),
            text_response("Loaded the ribeye. Step one: pat dry and season."),
        ],
    )
    turn = agent.respond("let's do the ribeye", language="en")
    assert turn.tool_calls == ["list_recipes", "load_recipe"]
    loaded = llm.calls[2]["messages"][-1]["content"][0]["content"]
    assert loaded.startswith("RECIPE LOADED: Pan-Seared Ribeye")


def test_express_and_sleep(settings, robot):
    slept = []
    agent, _, _ = make_agent(
        robot,
        settings,
        [
            tool_response("express", {"emotion": "success"}, tool_id="a"),
            tool_response("go_to_sleep", {}, tool_id="b"),
            text_response("Goodnight, chef."),
        ],
        sleep_flag=slept,
    )
    turn = agent.respond("that's all for tonight", language="en")
    assert "express:success" in robot.events
    assert slept == [True]
    assert turn.spoken == "Goodnight, chef."


def test_unknown_tool_and_failure_are_reported_not_raised(settings, robot):
    agent, llm, _ = make_agent(robot, settings, [tool_response("frobnicate", {}), text_response("ok")])
    agent.respond("x", language="en")
    assert "unknown tool" in llm.calls[1]["messages"][-1]["content"][0]["content"]

    agent2, llm2, _ = make_agent(
        robot, settings, [tool_response("set_timer", {"label": "x", "seconds": 0}), text_response("ok")]
    )
    agent2.respond("x", language="en")
    assert "failed" in llm2.calls[1]["messages"][-1]["content"][0]["content"]


def test_mandarin_hint_and_event_turns(settings, robot):
    agent, llm, _ = make_agent(
        robot, settings, [text_response("五分钟。"), text_response("Pasta timer is done.")]
    )
    agent.respond("要煮多久", language="zh")
    assert llm.calls[0]["messages"][0]["content"].endswith("(spoken in Mandarin)")
    turn = agent.announce("Timer 'pasta' has finished (9 minutes).")
    assert turn.spoken == "Pasta timer is done."
    assert llm.calls[1]["messages"][-1]["content"].startswith("[kitchen event]")


def test_history_trim_never_orphans_tool_results(settings, robot):
    responses = []
    for i in range(6):
        responses += [tool_response("nod", {}, tool_id=f"t{i}"), text_response(f"reply {i}")]
    agent, llm, _ = make_agent(robot, settings, responses)
    for i in range(6):
        agent.respond(f"turn {i}", language="en")
    msgs = agent.messages
    assert len(msgs) <= 8
    assert msgs[0]["role"] == "user" and isinstance(msgs[0]["content"], str)
    # every tool_result still has its tool_use in the preceding assistant message
    for i, m in enumerate(msgs):
        if m["role"] == "user" and isinstance(m["content"], list):
            ids = {b["tool_use_id"] for b in m["content"]}
            prev = msgs[i - 1]
            assert prev["role"] == "assistant"
            assert ids <= {b["id"] for b in prev["content"] if b.get("type") == "tool_use"}


def test_empty_turn_short_circuits(settings, robot):
    agent, llm, _ = make_agent(robot, settings, [])
    assert agent.respond("   ").spoken == ""
    assert llm.calls == []
