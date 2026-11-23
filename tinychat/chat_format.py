"""Frozen role tokens and label construction for Stage 1 Chinese chat."""

SPECIALS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>",
    "<|assistant|>", "<|end_turn|>",
)
SPECIAL_IDS = {token: index for index, token in enumerate(SPECIALS)}
DEFAULT_SYSTEM = "你是一个中文助手。请简洁、诚实地回答问题。"
IGNORE_LABEL = -100


def check_special_ids(tokenizer):
    actual = {token: tokenizer.token_to_id(token) for token in SPECIALS}
    if actual != SPECIAL_IDS:
        raise ValueError(f"Stage 1 special token IDs differ: {actual}")


def contains_reserved(text):
    return any(token in text for token in SPECIALS)


def manifest_final_eos(manifest):
    chat_format = manifest.get("chat_format", {})
    if not isinstance(chat_format, dict) or type(chat_format.get("final_eos", True)) is not bool:
        raise ValueError("Manifest chat_format.final_eos must be a boolean")
    return chat_format.get("final_eos", True)


def encode_conversation(tokenizer, messages, *, generation_prompt=False, include_system=True,
                        final_eos=True):
    """Return token IDs and unshifted labels, with assistant content supervised."""
    messages = list(messages)
    system = {"role": "system", "content": DEFAULT_SYSTEM}
    if messages and messages[0]["role"] == "system":
        system = messages.pop(0)

    expected_roles = ["user", "assistant"] * (len(messages) // 2)
    if generation_prompt:
        expected_roles.append("user")
    if not messages or [message["role"] for message in messages] != expected_roles:
        raise ValueError("Expected alternating user and assistant messages with the appropriate final role")

    ids = [SPECIAL_IDS["<|bos|>"]]
    labels = [IGNORE_LABEL]
    formatted = [system, *messages] if include_system else messages
    for message in formatted:
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or not content.strip() or contains_reserved(content):
            raise ValueError(f"Invalid or reserved content in {role} message")
        ids.append(SPECIAL_IDS[f"<|{role}|>"])
        labels.append(IGNORE_LABEL)
        turn = tokenizer.encode(content, add_special_tokens=False).ids + [SPECIAL_IDS["<|end_turn|>"]]
        ids.extend(turn)
        labels.extend(turn if role == "assistant" and not generation_prompt else [IGNORE_LABEL] * len(turn))

    if generation_prompt:
        ids.append(SPECIAL_IDS["<|assistant|>"])
        labels.append(IGNORE_LABEL)
    elif final_eos:
        eos = SPECIAL_IDS["<|eos|>"]
        ids.append(eos)
        labels.append(eos)

    return {"input_ids": ids, "labels": labels}
