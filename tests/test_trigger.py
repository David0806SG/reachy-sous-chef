from sous_chef.config import DEFAULT_NAME_ALIASES
from sous_chef.trigger import NameTrigger


def make(require_name=True, window=20.0):
    return NameTrigger(DEFAULT_NAME_ALIASES, follow_up_window_s=window, require_name=require_name)


def test_name_at_start_is_stripped():
    t = make()
    a = t.check("Reachy, how long for the sear?", now=100.0)
    assert a is not None and a.by_name
    assert a.text == "how long for the sear?"


def test_hey_prefix_and_whisper_spellings():
    t = make()
    for phrase in (
        "Hey Reachy set a timer",
        "Richie: set a timer",
        "ok ritchie set a timer",
        "Ricky set a timer",
    ):
        a = t.check(phrase, now=0.0)
        assert a is not None and a.by_name, phrase
        assert a.text == "set a timer", phrase


def test_mandarin_alias():
    t = make()
    a = t.check("瑞奇，帮我定五分钟", now=0.0)
    assert a is not None and a.by_name
    assert a.text == "帮我定五分钟"


def test_name_in_middle_does_not_count():
    t = make()
    assert t.check("I think Reachy is cute", now=0.0) is None


def test_follow_up_window():
    t = make(window=20.0)
    assert t.check("and the oven temperature?", now=0.0) is None
    t.open_window(now=0.0)
    a = t.check("and the oven temperature?", now=10.0)
    assert a is not None and not a.by_name and a.text == "and the oven temperature?"
    assert t.check("and the oven temperature?", now=25.0) is None
    t.open_window(now=30.0)
    t.close_window()
    assert t.check("anything", now=31.0) is None


def test_continuous_mode_answers_everything():
    t = make(require_name=False)
    a = t.check("what's the oven temp", now=0.0)
    assert a is not None and not a.by_name


def test_name_only_keeps_something_to_say():
    t = make()
    a = t.check("Reachy?", now=0.0)
    assert a is not None and a.text  # falls back to the raw utterance rather than empty
