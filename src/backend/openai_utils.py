import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, Dict

import openai
import requests
import tiktoken

from market_alerts.domain.constants import DEFAULT_MODEL_NAME, MODELS_WHITE__LIST

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_PROXY_URL", "https://ai-proxy.lab.epam.com")


def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_reply(reply, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(reply))


def get_openai_stream(messages, engine):
    if isinstance(messages, str):
        messages = [{"role": "assistant", "content": messages}]

    stream = openai.ChatCompletion.create(
        engine=engine, temperature=0, timeout=600, request_timeout=600, messages=messages, max_tokens=1024, stream=True
    )

    return stream


def get_openai_result(messages, engine, max_retries=3):
    error = ""
    if isinstance(messages, str):
        messages = [{"role": "assistant", "content": messages}]
    for i in range(max_retries):
        try:
            ret = openai.ChatCompletion.create(
                engine=engine, temperature=0, timeout=600, request_timeout=600, messages=messages, max_tokens=1024
            )
            return ret["choices"][0]["message"]["content"], ret["usage"]
        except openai.error.RateLimitError as e:
            error = e
            time.sleep(0.1)
            continue
    else:
        raise openai.error.RateLimitError(error)


def get_openai_stream_result(messages, engine, max_retries=3):
    error = ""
    for i in range(max_retries):
        try:
            stream = get_openai_stream(messages, engine)

            ret = ""
            for chunk in stream:
                ret += chunk["choices"][0]["delta"].get("content", "")

            prompt_tokens = num_tokens_from_messages(messages, engine)
            completion_tokens = num_tokens_from_reply(ret, engine)
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            return ret, token_usage
        except openai.error.RateLimitError as e:
            error = e
            time.sleep(0.1)
            continue
    else:
        raise openai.error.RateLimitError(error)


def make_history_context(prompt, dialogue, concat_prompt=""):
    messages = [
        {
            "role": "system",
            "content": prompt,
        }
    ]
    for i in range(len(dialogue) // 2):
        chat_user, chat_assistant = dialogue[2 * i], dialogue[2 * i + 1]
        messages.append({"role": "user", "content": chat_user})
        messages.append({"role": "assistant", "content": chat_assistant})
    messages.append({"role": "user", "content": " ".join([dialogue[-1], concat_prompt])})
    return messages


@lru_cache(maxsize=1)
def get_available_models() -> Dict[str, Any]:
    request_headers = {"API-KEY": openai.api_key}

    resp = requests.get(f"{openai.api_base}/openai/models", headers=request_headers)

    models = json.loads(resp.content)

    available_models = []

    for model in models["data"]:
        model_name = model.get("display_name")
        if model_name and model_name in MODELS_WHITE__LIST:
            if model_name == DEFAULT_MODEL_NAME:
                model["default_checked"] = True
            available_models.append(model)

    result = {m["display_name"]: (m["model"], m.get("default_checked", False)) for m in available_models}

    return result
