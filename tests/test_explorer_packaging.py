"""`web/index.html` can still serve `web/kernelsmith_explorer.jsx` (Task 13b).

The explorer is a single 73 KB JSX file with no build step: `index.html` fetches it,
strips the two module lines a browser cannot resolve, compiles it with Babel standalone,
and renders it. That works today. What this module protects is the part that will break
silently: the packaging makes **assumptions about the shape of the JSX**, and the JSX is
edited far more often than the HTML.

Three assumptions, each one a test:

1. exactly one real ESM import (React's), and stripping it removes exactly one line;
2. exactly one `export default function`, at column 0, which becomes the component;
3. the strip does not touch the Python inside `KERNEL_SOURCE`.

Assumption 3 is not hypothetical. The first version of the strip used
``/^\\s*import\\s+[^;]+;/gm``, and `[^;]+` crosses newlines: it ate 53 lines starting at
the `import torch` INSIDE the kernel-source template literal, and the page died with a
Babel parse error pointing at Python. The regexes are line-bound now, and these tests
apply the real ones — read out of `index.html`, not copied here, so a change to the HTML
is tested rather than assumed.

There is no Node in CI, so nothing here compiles the JSX. The compile-and-render path was
verified out-of-band with `@babel/standalone` + `react-dom/server`: 66 KB of HTML,
`<h1>gpuyantra</h1>`, and every measured number present.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
HTML = WEB / "index.html"
JSX = WEB / "kernelsmith_explorer.jsx"
DOCKERFILE = WEB / "Dockerfile"


@pytest.fixture(scope="module")
def html() -> str:
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def jsx() -> str:
    return JSX.read_text(encoding="utf-8")


def js_regex_from_html(html: str, marker: str) -> re.Pattern[str]:
    """Pull one `.replace(/.../m, ...)` pattern out of index.html as a Python regex.

    The point of reading it from the HTML rather than restating it: if someone loosens the
    regex in the page, these tests exercise the loosened one and fail there, instead of
    passing against a copy that no longer matches what ships.
    """
    line = next(row for row in html.splitlines() if marker in row)
    body = line[line.index("/") + 1 : line.rindex("/")]
    # JS and Python agree on everything these two patterns use.
    return re.compile(body, re.MULTILINE)


# --------------------------------------------------------------------------- #
# The files exist and are wired to each other
# --------------------------------------------------------------------------- #


def test_the_explorer_is_packaged_as_two_files_plus_a_dockerfile():
    for path in (HTML, JSX, DOCKERFILE):
        assert path.is_file(), f"{path} is part of the hosted explorer and must be committed"


def test_the_page_fetches_the_jsx_that_is_actually_committed(html: str):
    """One source of truth: no second copy of any measured number."""
    assert 'var SOURCE = "kernelsmith_explorer.jsx"' in html
    assert JSX.name in html


def test_the_page_is_titled_for_the_project_not_the_agent(html: str):
    """Task 13b: gpuyantra is the project, KernelSmith is the agent tree inside it."""
    assert "<title>gpuyantra — GPU Kernel Optimization</title>" in html


def test_react_and_babel_are_pinned(html: str):
    """A CDN shipping a new major React overnight is a demo that breaks while being judged."""
    for pin in ("react@18.3.1", "react-dom@18.3.1", "@babel/standalone@7.26.4"):
        assert pin in html, pin
    assert "@latest" not in html


def test_the_dockerfile_serves_both_files_on_the_port_cloud_run_uses(html: str):
    body = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY index.html" in body
    assert f"COPY {JSX.name}" in body
    assert "8080" in body
    # nginx has no MIME type for .jsx and would send application/octet-stream.
    assert "text/plain jsx" in body


# --------------------------------------------------------------------------- #
# The three assumptions the packaging makes about the JSX
# --------------------------------------------------------------------------- #


def test_the_jsx_has_exactly_one_esm_import_and_it_is_react(jsx: str):
    imports = re.findall(r"^import\s+[^\n]*from\s+['\"][^'\"]+['\"]", jsx, re.MULTILINE)
    assert len(imports) == 1, imports
    assert 'from "react"' in imports[0]


def test_the_jsx_has_exactly_one_default_export_at_column_zero(jsx: str):
    exports = re.findall(r"^export\s+[^\n]*", jsx, re.MULTILINE)
    assert len(exports) == 1, exports
    assert exports[0].startswith("export default function ")


def test_stripping_the_import_removes_exactly_one_line(html: str, jsx: str):
    """The bug this catches ate 53 lines and pointed Babel at Python."""
    pattern = js_regex_from_html(html, "import\\s+React")
    stripped = pattern.sub("", jsx, count=1)
    assert jsx.count("\n") - stripped.count("\n") == 1


def test_stripping_leaves_the_kernel_source_byte_for_byte(html: str, jsx: str):
    """The Python in `KERNEL_SOURCE` starts with `import torch` at column 0."""
    pattern = js_regex_from_html(html, "import\\s+React")
    stripped = pattern.sub("", jsx, count=1)

    start = jsx.index("const KERNEL_SOURCE = `")
    end = jsx.index("`;", start)
    kernel = jsx[start:end]
    assert "import torch" in kernel and "import triton.language as tl" in kernel
    assert kernel in stripped


def test_the_export_rewrite_produces_the_component_the_page_renders(html: str, jsx: str):
    pattern = js_regex_from_html(html, "export\\s+default")
    stripped = pattern.sub("var __EXPLORER__ = function \\1", jsx, count=1)
    assert "var __EXPLORER__ = function KernelSmithExplorer" in stripped
    assert not re.search(r"^export\s", stripped, re.MULTILINE)
    # ... which is exactly the name the page returns out of its factory.
    assert "return __EXPLORER__;" in html


def test_the_page_refuses_an_import_it_cannot_resolve(html: str):
    """A second import added to the JSX must fail loudly, not render half a page."""
    assert "has an import this page does not know how to resolve" in html


def test_the_hooks_the_component_imports_are_all_supplied_by_the_page(html: str, jsx: str):
    """The import is stripped, so every hook it named has to be passed in by hand."""
    imported = re.search(r"^import\s+React,\s*\{([^}]*)\}", jsx, re.MULTILINE)
    assert imported, "the React import no longer has the shape index.html expects"
    hooks = [name.strip() for name in imported.group(1).split(",") if name.strip()]
    factory = html[html.index("var factory = new Function(") : html.index("return __EXPLORER__")]
    for hook in hooks:
        assert f'"{hook}"' in factory, f"{hook} is imported by the JSX but not passed in"
        assert f"React.{hook}" in html, f"{hook} is declared but never bound"
