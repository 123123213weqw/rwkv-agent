"""Pure prompt builders for semantic Agent routing."""

from __future__ import annotations


def render_tool_gate_prompt(
    message: str,
    *,
    context: str = "",
    has_pasted_text: bool = False,
) -> str:
    """Classify retrieval need semantically instead of matching topic words."""

    prompt = (
        "System: Classify only the final Current user request. Reply with exactly "
        "one lowercase label: search or chat. Decide from meaning, not from the "
        "presence of words such as search, author, company, version, or source. "
        "Use search whenever a reliable answer requires retrieving an external "
        "fact: entity-specific people, authors, companies, organizations, dates, "
        "versions, prices, policies, statistics, papers, products, public events, "
        "live information, explicit sources, local/private indexed knowledge, or "
        "facts from the active pasted text. This remains search even when the user "
        "does not explicitly ask to search. Use chat only for greetings, casual "
        "conversation, writing, translation, arithmetic, coding, general concept "
        "explanations, hypothetical transformations, or reasoning fully supported "
        "by text already present in the request. A quoted factual question being "
        "translated or rewritten is chat, not search. Recent conversation is only "
        "reference context for pronouns and follow-ups; never classify an earlier "
        "request instead of the Current user request.\n\n"
        "Current user request: Hello.\nAssistant: chat\n\n"
        "Current user request: Write a Python function that reverses a string."
        "\nAssistant: chat\n\n"
        "Current user request: Explain recursion in simple terms."
        "\nAssistant: chat\n\n"
        "Current user request: 把“Mamba架构是谁提出的”翻译成英文。"
        "\nAssistant: chat\n\n"
        "Current user request: Mamba架构最初是谁提出的？\nAssistant: search\n\n"
        "Current user request: Node.js当前LTS版本是什么？"
        "\nAssistant: search\n\n"
        "Current user request: Linux内核维护者和Linux基金会是什么关系？"
        "\nAssistant: search\n\n"
        "Current user request: Search the local knowledge base for the scheduler "
        "design.\nAssistant: search\n\n"
        "System: Recent conversation reference:\nUser: 我们正在讨论Mamba架构。"
        "\nAssistant: 好的。\nEnd recent conversation.\n"
        "Current user request: 那它是谁创建的？\nAssistant: search\n\n"
        "System: Recent conversation reference:\nUser: We are discussing the "
        "JAX library.\nAssistant: Understood.\nEnd recent conversation.\n"
        "Current user request: Who originally created it?\nAssistant: search\n\n"
        "System: Active pasted long text: yes.\n"
        "Current user request: 材料中的项目负责人是谁？\nAssistant: search\n\n"
        "System: Active pasted long text: yes.\n"
        "Current user request: Summarize the pasted document.\nAssistant: search\n\n"
        "System: Active pasted long text: yes.\n"
        "Current user request: 谢谢。\nAssistant: chat\n\n"
    )
    if context.strip():
        prompt += (
            "System: Recent conversation reference:\n"
            + context.strip()
            + "\nEnd recent conversation.\n\n"
        )
    prompt += (
        "System: Active pasted long text: "
        + ("yes" if has_pasted_text else "no")
        + ". Its presence alone does not force search; an unrelated greeting is "
        "still chat.\n"
        + f"Current user request: {message.strip()}\nAssistant: "
    )
    return prompt
