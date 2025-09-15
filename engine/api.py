# import os, json, shutil
# from openai import OpenAI
# from pathlib import Path

# import logging
# logger = logging.getLogger("ai_pokemon_trainer")

# BASE_DIR = Path(__file__).resolve().parent.parent

# SECRET_SETTING = None
# if not os.path.exists(BASE_DIR / 'secret_setting.json'):
#     shutil.copyfile(BASE_DIR / 'secret_setting.json.example',
#                     BASE_DIR / 'secret_setting.json')
# with open(BASE_DIR / 'secret_setting.json', 'r') as fp:
#     SECRET_SETTING = json.loads(fp.read())

# client = OpenAI(
#     api_key=SECRET_SETTING["api-key"],
#     base_url=SECRET_SETTING["base-url"],
# )

# def get_ai_response(prompt, cnt=1):
#     logger.debug(f"Send to API, {json.dumps(prompt, indent=4, separators=(',', ': '), ensure_ascii=False)}")
#     try:
#         response = client.chat.completions.create(
#             model=SECRET_SETTING["model"],
#             messages=prompt,
#             response_format={"type": "json_object"},
#         )
#     except Exception as e:
#         if cnt>3:
#             raise BaseException(f"Request API ERROR, and ther is no pssibility of countinue. STOP!")
#         logger.error(f"Request API ERROR: {e}, Retry {cnt}!")
#         return get_ai_response(prompt, cnt+1)
#     logger.debug(f"Recived by API, {json.dumps(response.choices[0].message.content, indent=4, separators=(',', ': '), ensure_ascii=False)}")
#     logger.info(f"API token usage: {response.usage.total_tokens}")
#     return response.choices[0].message.content, response.usage.total_tokens

# api.py
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

# --------------------------------------------------------------------------------------
# Load settings
# --------------------------------------------------------------------------------------
def _load_secret_settings(path: str = "secret_setting.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Copy secret_setting.json.example to {path} and edit values."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SECRET_SETTING: Dict[str, Any] = _load_secret_settings()

PROVIDER: str = (SECRET_SETTING.get("provider") or "").lower()
API_KEY: str = SECRET_SETTING.get("api-key") or "not-needed"
BASE_URL: str = SECRET_SETTING.get("base-url") or "http://localhost:11434/v1"
MODEL: str = SECRET_SETTING.get("model") or "llama3.1:8b"
JSON_MODE: bool = bool(SECRET_SETTING.get("json_mode", False))

# --------------------------------------------------------------------------------------
# OpenAI-compatible client
# --------------------------------------------------------------------------------------
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=180,
)

# --------------------------------------------------------------------------------------
# Core chat function
# --------------------------------------------------------------------------------------
def get_ai_response(messages: List[Dict[str, str]], cnt: int = 1) -> Tuple[str, int]:
    """
    Send a chat completion request to the configured provider.

    Args:
        messages: [{"role": "user"/"system"/"assistant", "content": "..."}]
        cnt: retry counter

    Returns:
        (content, total_tokens)
    """
    try:
        logger.debug("Send to API: %s", json.dumps(messages, ensure_ascii=False))
    except Exception:
        logger.debug("Send to API: <unserializable messages>")

    kwargs: Dict[str, Any] = dict(
        model=MODEL,
        messages=messages,
    )

    # JSON mode per provider
    if JSON_MODE:
        if PROVIDER in ("openai", ""):
            kwargs["response_format"] = {"type": "json_object"}
        elif PROVIDER in ("ollama", "lm-studio", "llamacpp"):
            kwargs["extra_body"] = {"format": "json"}
        else:
            kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        if JSON_MODE and cnt <= 1:
            logger.warning("JSON mode failed (%s). Retrying without JSON mode...", e)
            kwargs.pop("response_format", None)
            kwargs.pop("extra_body", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e2:
                if cnt > 3:
                    raise BaseException("Request API ERROR, no possibility to continue.") from e2
                logger.error("API ERROR after JSON fallback: %s, Retry %d!", e2, cnt)
                return get_ai_response(messages, cnt + 1)
        else:
            if cnt > 3:
                raise BaseException("Request API ERROR, no possibility to continue.") from e
            logger.error("API ERROR: %s, Retry %d!", e, cnt)
            return get_ai_response(messages, cnt + 1)

    # Extract content
    content = ""
    try:
        content = response.choices[0].message.content
    except Exception:
        pass

    try:
        logger.debug("Received by API: %s", json.dumps(content, ensure_ascii=False))
    except Exception:
        logger.debug("Received by API: <unserializable content>")

    # Token usage (may be absent for local providers)
    total_tokens = 0
    try:
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "total_tokens", None) is not None:
            total_tokens = int(usage.total_tokens)
            logger.info("API token usage: %d", total_tokens)
        else:
            logger.info("API token usage: (not provided by this provider)")
    except Exception:
        logger.info("API token usage: (not available)")

    return content or "", total_tokens


# --------------------------------------------------------------------------------------
# CLI test
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(
        "Provider=%s | Base URL=%s | Model=%s | JSON mode=%s",
        PROVIDER, BASE_URL, MODEL, JSON_MODE
    )
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply in one short sentence."},
        {"role": "user", "content": "Say hello and tell me which model you are."}
    ]
    out, toks = get_ai_response(test_messages)
    print("\n--- RESPONSE ---\n", out)
    print("\n--- TOKENS ---\n", toks)

