from __future__ import annotations

import json

from rwkv_search.realtime.internal_site_search import (
    build_aop_search_payload,
    build_html_form_payload,
    build_internal_search_queries,
    extract_internal_search_forms,
    parse_aop_search_results,
)


def test_detects_webplus_capability_from_public_form_semantics() -> None:
    html = b"""
    <form method="post"
      action="/_web/_search/api/search/new.rst?_p=opaque"
      portletmode="search">
      <input name="context" type="hidden" value="">
      <input name="keyword" type="text" placeholder="search">
      <input name="submit" type="submit">
    </form>
    """

    forms = extract_internal_search_forms(
        html,
        base_url="https://cs.example.edu.cn/",
    )

    assert len(forms) == 1
    assert forms[0].protocol == "webplus_ajax"
    assert forms[0].query_field == "keyword"
    assert forms[0].hidden_fields == {"context": ""}


def test_rejects_cross_origin_or_non_search_forms() -> None:
    html = b"""
    <form action="https://tracker.invalid/search">
      <input name="keyword" type="text">
    </form>
    <form action="/contact"><input name="name" type="text"></form>
    """

    assert not extract_internal_search_forms(
        html,
        base_url="https://example.org/",
    )


def test_detects_vsb_lucene_hidden_query_protocol() -> None:
    html = b"""
    <form action="ssjg.jsp?wbtreeid=1001" method="post" class="search-box">
      <input name="lucenenewssearchkey" type="hidden" value="">
      <input name="_lucenesearchtype" type="hidden" value="1">
      <input name="showkeycode" type="text">
    </form>
    """

    forms = extract_internal_search_forms(
        html,
        base_url="https://cs.example.edu.cn/",
    )

    assert len(forms) == 1
    assert forms[0].protocol == "vsb_lucene"
    assert forms[0].query_field == "showkeycode"
    assert build_html_form_payload(forms[0], "Calton Pu") == {
        "lucenenewssearchkey": "Calton Pu",
        "_lucenesearchtype": "1",
        "showkeycode": "Calton Pu",
    }


def test_detects_formless_webplus_search_url_declaration() -> None:
    html = b"""
    <div class="wp_search">
      <input id="keyword" name="keyword" type="text">
      <input type="hidden" id="securl"
        value="/_web/_search/api/search/new.rst?locale=zh_CN&amp;_p=opaque">
      <button onclick="checkValues()">search</button>
    </div>
    """

    forms = extract_internal_search_forms(
        html,
        base_url="https://cs.example.edu.cn/",
    )

    assert len(forms) == 1
    assert forms[0].protocol == "webplus_ajax"
    assert forms[0].method == "get"
    assert forms[0].query_field == "keyword"
    assert forms[0].action.startswith(
        "https://cs.example.edu.cn/_web/_search/api/search/new.rst"
    )


def test_internal_queries_prefer_user_exact_terms_and_strip_site_operator() -> None:
    queries = build_internal_search_queries(
        "谁主持了“全民禁毒宣传教育”主题班会？",
        ("全民禁毒宣传教育 主持人 site:cs.example.edu.cn",),
        max_queries=3,
    )

    assert queries[0] == "全民禁毒宣传教育"
    assert all("site:" not in value.casefold() for value in queries)
    assert len(queries) <= 3


def test_detects_script_declared_aop_search_capability() -> None:
    html = b"""
    <input name="showkeycode" v-model="query.keyWord">
    <script>
      var appOwner = "2125615756";
      var urlPrefix = "/aop_component/";
      var url = "/aop_views/search/modules/resultpc/soso.html";
    </script>
    """

    forms = extract_internal_search_forms(
        html,
        base_url="https://cs.example.edu.cn/",
    )

    assert len(forms) == 1
    assert forms[0].protocol == "aop_search"
    assert forms[0].action == (
        "https://cs.example.edu.cn/aop_component/webber/search/search/search/queryPage"
    )
    payload = build_aop_search_payload(forms[0], "全民禁毒宣传教育")
    assert payload["owner"] == "2125615756"
    assert payload["token"] == "tourist"
    assert payload["keyWord"] == "全民禁毒宣传教育"
    assert payload["page"] == {"current": 0, "size": 20}


def test_parses_aop_records_and_rejects_cross_site_urls() -> None:
    body = json.dumps(
        {
            "code": "0000",
            "data": {
                "page": {
                    "records": [
                        {
                            "url": "//cs.example.edu.cn/info/1013/1096.htm",
                            "title": "<span>全民禁毒宣传教育</span>主题班会",
                            "intro": "活动由班长主持。",
                        },
                        {
                            "url": "https://outside.invalid/nope",
                            "title": "outside",
                        },
                    ]
                }
            },
        },
        ensure_ascii=False,
    ).encode()

    results = parse_aop_search_results(
        body,
        base_url="https://cs.example.edu.cn/aop_component/search",
        scope_host="cs.example.edu.cn",
        query="全民禁毒宣传教育",
    )

    assert [(item.url, item.title) for item in results] == [
        (
            "https://cs.example.edu.cn/info/1013/1096.htm",
            "全民禁毒宣传教育 主题班会",
        )
    ]
