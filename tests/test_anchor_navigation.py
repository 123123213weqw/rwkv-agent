from __future__ import annotations

import asyncio

from rwkv_search.realtime.anchor_navigation import (
    AnchorPageCache,
    extract_anchor_links,
    select_navigation_links,
)


def test_extracts_anchor_text_context_and_filters_non_content() -> None:
    html = b"""
    <main>
      <article><h2>Release archive</h2>
        <p>Stable Python release notes
          <a href="/downloads/release/python-399/">Python 3.9.9</a>
        </p>
      </article>
      <a href="https://outside.invalid/page">outside</a>
      <a href="/assets/manual.pdf">asset</a>
      <a href="/login">login</a>
    </main>
    """

    links = extract_anchor_links(
        html,
        base_url="https://www.example.org/archive/",
    )

    assert [link.url for link in links] == [
        "https://www.example.org/downloads/release/python-399"
    ]
    assert links[0].title == "Python 3.9.9"
    assert "Stable Python release notes" in links[0].context


def test_anchor_context_drives_selection_when_url_is_opaque() -> None:
    html = """
    <ul>
      <li>Campus navigation <a href="/p/100">Directory</a></li>
      <li>Calton Pu wins test-of-time award <a href="/p/200">Read more</a></li>
      <li>Student sports schedule <a href="/p/300">Read more</a></li>
    </ul>
    """
    links = extract_anchor_links(html, base_url="https://example.edu/")

    selected = select_navigation_links(
        "Which award did Calton Pu receive?",
        ("Calton Pu award",),
        links,
        max_links=1,
    )

    assert [link.url for link in selected] == ["https://example.edu/p/200"]


def test_rel_next_is_retained_as_bounded_pagination_control() -> None:
    html = """
    <a href="/news/a">Unrelated archive item</a>
    <a rel="next" href="/news?page=2">Next</a>
    """
    links = extract_anchor_links(html, base_url="https://example.org/news")

    selected = select_navigation_links(
        "target announcement",
        (),
        links,
        max_links=2,
        max_pagination_links=1,
    )

    assert any(link.pagination and "page=2" in link.url for link in selected)


def test_page_cache_singleflights_duplicate_urls() -> None:
    class Content:
        async def read(self, _: int) -> bytes:
            await asyncio.sleep(0.01)
            return b"<html>ok</html>"

    class Response:
        status = 200
        url = "https://example.org/"
        headers = {"Content-Type": "text/html"}
        content = Content()

        async def __aenter__(self) -> Response:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    class Session:
        calls = 0

        def get(self, *_: object, **__: object) -> Response:
            self.calls += 1
            return Response()

    async def run() -> tuple[tuple[object, bool], tuple[object, bool], int]:
        session = Session()
        cache = AnchorPageCache()
        first, second = await asyncio.gather(
            cache.fetch(
                session,
                "https://example.org/",
                scope_host="example.org",
                timeout_seconds=1,
                max_body_bytes=1000,
            ),
            cache.fetch(
                session,
                "https://example.org/",
                scope_host="example.org",
                timeout_seconds=1,
                max_body_bytes=1000,
            ),
        )
        return first, second, session.calls

    first, second, calls = asyncio.run(run())

    assert calls == 1
    assert {first[1], second[1]} == {False, True}
