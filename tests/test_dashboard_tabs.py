"""The dashboard's sidebar navigation renders, and the Optimize tab still works.

`streamlit_app.py` runs `main()` at import, so it cannot be imported directly — it is
driven through `AppTest`, which executes the script the way the server would.

Two things are worth locking down about the tabs added in Task 10, and neither had any
coverage before:

1. **All three views render without an exception.** The dashboard is the demo surface; a
   traceback on a tab switch is a blank page in front of an audience. This is also the
   first automated check that the page renders at all — until now that was verified by
   opening it.

2. **The Optimize tab is unchanged.** The two-message protocol is driven by
   `drive_run()` from inside `main()`, on every rerun and on every tab, so that a run
   started on Optimize still gets its follow-up message if the operator wanders off to
   the audit view. A tab guard placed one line too early would silently break the
   hot-swap while leaving every panel looking fine.

Nothing here needs a GPU, Vertex credentials, or the network: no audit is run (the tab is
asserted in its pre-click state) and the Skill Library tab degrades to a warning when
Firestore is unreachable, which is a rendered state either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: Absolute, because `AppTest.from_file` resolves a relative path against the file that
#: calls it — which is this one, inside tests/.
APP = Path(__file__).resolve().parent.parent / "kernelsmith" / "ui" / "streamlit_app.py"
TIMEOUT_S = 120


@pytest.fixture
def app():
    """A freshly run dashboard, or a skip if this streamlit build has no AppTest."""
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    at.run()
    return at


def exceptions(at) -> list[str]:
    return [str(element.value) for element in at.exception]


def test_the_dashboard_renders_at_all(app):
    assert exceptions(app) == []
    assert app.title[0].value == "⚒️ KernelSmith"


def test_the_sidebar_offers_the_three_views(app):
    from kernelsmith.ui.streamlit_app import TABS

    assert app.sidebar.radio[0].options == list(TABS)


def test_the_default_view_is_optimize_and_it_still_has_its_start_button(app):
    from kernelsmith.ui.streamlit_app import TAB_OPTIMIZE

    assert app.sidebar.radio[0].value == TAB_OPTIMIZE
    assert "🚀 Start Optimization" in [button.label for button in app.button]


def test_the_audit_tab_renders_and_offers_every_registered_model(app):
    from kernelsmith.config import MODEL_REGISTRY
    from kernelsmith.ui.streamlit_app import TAB_AUDIT

    app.sidebar.radio[0].set_value(TAB_AUDIT).run()

    assert exceptions(app) == []
    model_options = app.selectbox[0].options
    assert len(model_options) == len(MODEL_REGISTRY)
    for key in MODEL_REGISTRY:
        assert any(option.startswith(f"{key} —") for option in model_options)
    assert app.selectbox[1].options == ["cpu", "cuda"]
    assert "Run Audit" in [button.label for button in app.button]


def test_the_audit_tab_says_nothing_until_an_audit_is_run(app):
    """No fabricated table, and no audit kicked off by a 1 Hz refresh."""
    from kernelsmith.ui.streamlit_app import TAB_AUDIT

    app.sidebar.radio[0].set_value(TAB_AUDIT).run()

    assert app.dataframe.values == []
    assert any("No audit yet" in info.value for info in app.info)


def test_the_skill_library_tab_renders(app):
    """Either a table or a stated reason there is none — never a traceback."""
    from kernelsmith.ui.streamlit_app import TAB_LIBRARY

    app.sidebar.radio[0].set_value(TAB_LIBRARY).run()

    assert exceptions(app) == []
    rendered = bool(app.dataframe.values) or bool(app.info.values) or bool(app.warning.values)
    assert rendered, "the skill library tab rendered neither a table nor an explanation"


def test_switching_back_to_optimize_restores_the_three_column_dashboard(app):
    from kernelsmith.ui.streamlit_app import TAB_AUDIT, TAB_OPTIMIZE

    app.sidebar.radio[0].set_value(TAB_AUDIT).run()
    app.sidebar.radio[0].set_value(TAB_OPTIMIZE).run()

    assert exceptions(app) == []
    assert "🚀 Start Optimization" in [button.label for button in app.button]
    assert any("Agent Activity" in header.value for header in app.subheader)


def test_the_run_driver_is_called_on_every_tab_not_only_on_optimize():
    """The follow-up message must be sent even if the operator switched views.

    Asserted on the source rather than by driving a real run: turn 2 needs Vertex, a GPU
    and a live inference server. What can rot silently is the ORDER of the tab guard —
    `drive_run()` moving below the `return`s would break the hot-swap and leave every
    panel looking healthy, which is precisely the failure mode this project treats as
    worse than a crash.
    """
    source = APP.read_text()
    body = source.split("def main() -> None:", 1)[1]

    assert "drive_run(consumer)" in body
    assert body.index("drive_run(consumer)") < body.index("if tab == TAB_AUDIT:"), (
        "drive_run() must run before the tab dispatch returns, or turn 2 is never sent"
    )
