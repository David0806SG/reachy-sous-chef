import threading
import time

from sous_chef.kitchen.recipes import RecipeBook
from sous_chef.kitchen.timers import TimerManager, format_duration


def test_format_duration():
    assert format_duration(45) == "45 seconds"
    assert format_duration(60) == "1 minute"
    assert format_duration(150) == "2 minutes 30 seconds"
    assert format_duration(3600) == "1 hour"
    assert format_duration(0) == "0 seconds"


def test_timer_fires_and_reports():
    fired = []
    done = threading.Event()

    def on_done(t):
        fired.append(t.label)
        done.set()

    tm = TimerManager(on_done)
    t = tm.start("pasta", 0.2)
    assert t.remaining > 0
    assert "pasta" in tm.describe()
    assert done.wait(2.0)
    assert fired == ["pasta"]
    assert tm.describe() == "No timers running."


def test_timer_cancel_and_replace():
    fired = []
    tm = TimerManager(lambda t: fired.append(t.label))
    tm.start("eggs", 0.3)
    assert tm.cancel("eggs")
    assert not tm.cancel("eggs")
    tm.start("rest", 5)
    tm.start("rest", 0.1)  # replaces
    time.sleep(0.4)
    assert fired == ["rest"]
    tm.start("a", 5)
    tm.start("b", 5)
    tm.cancel_all()
    assert tm.active() == []


def test_recipe_book(tmp_path):
    (tmp_path / "hainanese-chicken-rice.md").write_text(
        "# Hainanese Chicken Rice\n\n1. Poach.\n", encoding="utf-8"
    )
    (tmp_path / "pan-seared-ribeye.md").write_text("# Pan-Seared Ribeye\n\n1. Sear.\n", encoding="utf-8")
    book = RecipeBook(tmp_path)
    assert book.list() == [
        ("hainanese-chicken-rice", "Hainanese Chicken Rice"),
        ("pan-seared-ribeye", "Pan-Seared Ribeye"),
    ]
    assert book.find("chicken rice").slug == "hainanese-chicken-rice"
    assert book.find("RIBEYE").title == "Pan-Seared Ribeye"
    assert book.find("pan-seared-ribeye").body.startswith("# Pan-Seared")
    assert book.find("laksa") is None
    assert book.find("") is None


def test_recipe_book_missing_folder(tmp_path):
    assert RecipeBook(tmp_path / "nope").list() == []
