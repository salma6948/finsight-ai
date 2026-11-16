"""Small, local-only Ollama client. No external LLM service or native ML packages."""
from __future__ import annotations
import ipaddress
import os
from urllib.parse import urlparse
import requests


def base_url() -> str:
    url = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.username or parsed.password or parsed.path not in ('', '/'): 
        raise ValueError('Ollama must use a plain local HTTP address')
    try:
        if not ipaddress.ip_address(parsed.hostname or '').is_loopback:
            raise ValueError('Ollama must run on localhost')
    except ValueError as exc:
        if parsed.hostname != 'localhost':
            raise ValueError('Ollama must run on localhost') from exc
    return url


def request(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    response = requests.post(f'{base_url()}/api/{endpoint}', json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    model = model or os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
    result = []
    for i in range(0, len(texts), 24):
        batch = texts[i:i + 24]
        vectors = request('embed', {'model': model, 'input': batch, 'truncate': True}, timeout=120)['embeddings']
        if len(vectors) != len(batch) or not all(vectors):
            raise RuntimeError('Ollama embedding response length mismatch')
        result.extend(vectors)
    return result


def chat(system: str, user: str, model: str | None = None) -> str:
    model = model or os.getenv('OLLAMA_CHAT_MODEL', 'qwen3:4b')
    response = request('chat', {
        'model': model, 'stream': False, 'think': False,
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        'options': {'temperature': 0, 'num_predict': 650}
    }, timeout=180)
    return str(response['message'].get('content', '')).strip()


def available() -> bool:
    try:
        return bool(request('show', {'model': os.getenv('OLLAMA_CHAT_MODEL', 'qwen3:4b')}, timeout=3))
    except (requests.RequestException, ValueError, KeyError):
        return False
