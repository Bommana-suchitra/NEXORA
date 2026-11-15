"""
LLM backends: vLLM (in-process) and OpenRouter.

A backend takes a batch of chat requests and returns one raw completion per request, constrained
to the request's pydantic schema (vLLM: structured outputs; OpenRouter: `json_schema` response
format). `generate` parses and validates the completions and retries the ones that fail.
`batched` says how `simulate` should drive a backend: in rounds of one large batch (in-process
vLLM), or one request at a time per episode, so no episode waits for another (HTTP backends).
"""

import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any, Optional

from pydantic import BaseModel


class GenerationError(RuntimeError):
    """A request produced no valid output after all attempts."""


@dataclass
class Request:
    messages: list[dict[str, str]]
    output_type: type[BaseModel]
    sampling: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


@cache
def _schema(output_type: type[BaseModel]) -> dict:
    return output_type.model_json_schema()


def _json_span(text: str) -> str:
    """Strip anything around the outermost JSON object (e.g. markdown fences) or parse key-value lines."""
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1]
    import json, re
    data = {}
    for line in text.strip().splitlines():
        line_clean = line.strip()
        m = re.match(r"^\s*[\*\-#\s]*([a-zA-Z_]+)[\*\s]*:\s*(.*)$", line_clean)
        if m:
            k = m.group(1).lower().strip()
            v = m.group(2).strip()
            v = re.sub(r"^[\*\"'_`\s]+|[\*\"'_`\s]+$", "", v)
            if v.isdigit():
                data[k] = int(v)
            elif v.lower() in ("none", "null", ""):
                data[k] = None
            elif v.startswith("[") and v.endswith("]"):
                try:
                    data[k] = json.loads(v)
                except Exception:
                    inner = v[1:-1].strip()
                    data[k] = [re.sub(r"^[\*\"'_`\s]+|[\*\"'_`\s]+$", "", x) for x in inner.split(",") if x.strip()] if inner else []
            else:
                data[k] = v
    if data:
        data.setdefault("understood", data.get("explanation") or "Understood patient report.")
        data.setdefault("red_flags", [])
        data.setdefault("vital", None)
        return json.dumps(data)
    return text


class VLLM:
    """In-process vLLM engine. Each call to `complete` is one batched `LLM.chat`.

    Args:
        model: local path or Hugging Face id.
        chat_template_kwargs: passed to the chat template, e.g. {"enable_thinking": False}.
        **engine_args: passed to `vllm.LLM`, e.g. max_model_len=8192, gpu_memory_utilization=0.9.
    """

    batched = True

    def __init__(self, model: str, *, chat_template_kwargs: Optional[dict] = None, **engine_args):
        from vllm import LLM

        self.model = model
        self.chat_template_kwargs = chat_template_kwargs
        self.llm = LLM(model=model, **engine_args)
        # the model's generation_config (temperature, top_p, ...), as `vllm serve` would use
        self.defaults = self.llm.model_config.get_diff_sampling_param()
        self.usage: Counter = Counter()

    def stream_chat(self, messages: list[dict], max_new_tokens: int = 200):
        """Token-level progressive streaming chat completions using Hugging Face TextIteratorStreamer."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        thread = Thread(target=self.llm.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token

    def complete(self, requests: list[Request]) -> list[str | Exception]:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        params = [
            SamplingParams(
                **{**self.defaults, **r.sampling},
                seed=r.seed,
                structured_outputs=StructuredOutputsParams(
                    json=_schema(r.output_type), disable_any_whitespace=True
                ),
            )
            for r in requests
        ]
        try:
            outputs = self.llm.chat(
                [r.messages for r in requests],
                params,
                use_tqdm=False,
                chat_template_kwargs=self.chat_template_kwargs,
            )
        except Exception as e:
            # one bad request (e.g. a prompt longer than max_model_len) fails the whole batch
            if len(requests) == 1:
                return [e]
            return [out for r in requests for out in self.complete([r])]

        results: list[str | Exception] = []
        for out, p in zip(outputs, params):
            completion = out.outputs[0]
            self.usage.update(
                calls=1, prompt_tokens=len(out.prompt_token_ids), completion_tokens=len(completion.token_ids)
            )
            if completion.finish_reason == "length":
                results.append(GenerationError(f"output truncated at max_tokens={p.max_tokens}"))
            else:
                results.append(completion.text)
        return results


# chat.completions.create arguments; any other sampling key (top_k, min_p, ...) goes in extra_body
_OPENAI_PARAMS = {"temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop"}


class OpenRouter:
    """OpenRouter (or any OpenAI-compatible server, e.g. `vllm serve`) chat completions, at most
    `max_workers` requests in flight at a time across all calls to `complete`.

    The API key comes from `api_key` or OPENROUTER_API_KEY (environment or a .env file). The
    default `extra_body` asks reasoning models for low reasoning effort.
    """

    batched = False

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_workers: int = 16,
        extra_body: Optional[dict] = None,
    ):
        from dotenv import find_dotenv, load_dotenv
        from openai import OpenAI

        load_dotenv(find_dotenv(usecwd=True))
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter needs an API key: set OPENROUTER_API_KEY or pass api_key")

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)
        self.max_workers = max_workers
        self.extra_body = {"reasoning": {"effort": "low"}} if extra_body is None else extra_body
        self.usage: Counter = Counter()
        self._slots = threading.Semaphore(max_workers)  # `complete` may be called from many threads
        self._usage_lock = threading.Lock()

    def _complete_one(self, r: Request) -> tuple[str | Exception, Counter]:
        kwargs = {k: v for k, v in r.sampling.items() if k in _OPENAI_PARAMS}
        extra = {k: v for k, v in r.sampling.items() if k not in _OPENAI_PARAMS}
        if r.seed is not None:
            kwargs["seed"] = r.seed
        usage = Counter()
        try:
            with self._slots:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=r.messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": r.output_type.__name__, "strict": True, "schema": _schema(r.output_type)},
                    },
                    extra_body={**self.extra_body, **extra},
                    **kwargs,
                )
            usage.update(calls=1)
            if resp.usage:
                usage.update(prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)
            choice = resp.choices[0]  # error payloads can come back without choices
            if choice.finish_reason == "length":
                return GenerationError(f"output truncated at max_tokens={kwargs.get('max_tokens')}"), usage
            return choice.message.content or "", usage
        except Exception as e:
            return e, usage

    def stream_chat(self, messages: list[dict], max_new_tokens: int = 200):
        """Token-level progressive streaming chat completions using Hugging Face TextIteratorStreamer."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        thread = Thread(target=self.llm.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token

    def complete(self, requests: list[Request]) -> list[str | Exception]:
        with ThreadPoolExecutor(self.max_workers) as pool:
            done = list(pool.map(self._complete_one, requests))
        with self._usage_lock:
            for _, usage in done:
                self.usage.update(usage)
        return [text for text, _ in done]


class HuggingFace:
    """Hugging Face Inference API / Router (OpenAI-compatible) chat completions.

    Reads HF_TOKEN or HUGGING_FACE_HUB_TOKEN from .env or environment.
    """

    batched = False

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://router.huggingface.co/hf-inference/v1",
        max_workers: int = 8,
        extra_body: Optional[dict] = None,
    ):
        from dotenv import find_dotenv, load_dotenv
        from openai import OpenAI

        load_dotenv(find_dotenv(usecwd=True))
        api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not api_key:
            raise ValueError("Hugging Face backend needs a token: set HF_TOKEN in .env or pass api_key")

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)
        self.max_workers = max_workers
        self.extra_body = {} if extra_body is None else extra_body
        self.usage: Counter = Counter()
        self._slots = threading.Semaphore(max_workers)
        self._usage_lock = threading.Lock()

    def _complete_one(self, r: Request) -> tuple[str | Exception, Counter]:
        kwargs = {k: v for k, v in r.sampling.items() if k in _OPENAI_PARAMS}
        extra = {k: v for k, v in r.sampling.items() if k not in _OPENAI_PARAMS}
        if r.seed is not None:
            kwargs["seed"] = r.seed
        usage = Counter()
        try:
            with self._slots:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=r.messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": r.output_type.__name__, "strict": True, "schema": _schema(r.output_type)},
                        },
                        extra_body={**self.extra_body, **extra},
                        **kwargs,
                    )
                except Exception:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=r.messages,
                        response_format={"type": "json_object"},
                        extra_body={**self.extra_body, **extra},
                        **kwargs,
                    )
            usage.update(calls=1)
            if resp.usage:
                usage.update(prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                return GenerationError(f"output truncated at max_tokens={kwargs.get('max_tokens')}"), usage
            return choice.message.content or "", usage
        except Exception as e:
            return e, usage

    def stream_chat(self, messages: list[dict], max_new_tokens: int = 200):
        """Token-level progressive streaming chat completions using Hugging Face TextIteratorStreamer."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        thread = Thread(target=self.llm.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token

    def complete(self, requests: list[Request]) -> list[str | Exception]:
        with ThreadPoolExecutor(self.max_workers) as pool:
            done = list(pool.map(self._complete_one, requests))
        with self._usage_lock:
            for _, usage in done:
                self.usage.update(usage)
        return [text for text, _ in done]


class TransformersLocal:
    """Local Hugging Face transformers backend on GPU (NVIDIA CUDA) or CPU."""

    batched = False

    def __init__(self, model: str, *, device: str = "cuda", **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.llm = AutoModelForCausalLM.from_pretrained(
            model,
            token=token,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            **kwargs,
        )
        self.llm.eval()
        self.usage = Counter()

    def stream_chat(self, messages: list[dict], max_new_tokens: int = 200):
        """Token-level progressive streaming chat completions using Hugging Face TextIteratorStreamer."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        thread = Thread(target=self.llm.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token

    def complete(self, requests: list[Request]) -> list[str | Exception]:
        import torch
        results = []
        for r in requests:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    r.messages, tokenize=False, add_generation_prompt=True
                )
                prompt_text += "{"
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.llm.device)
                input_len = inputs["input_ids"].shape[1]

                gen_kwargs = {
                    "max_new_tokens": min(r.sampling.get("max_tokens", 350), 380),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "use_cache": True,
                }
                temp = r.sampling.get("temperature", 0.0)
                if temp and temp > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temp
                    if "top_p" in r.sampling:
                        gen_kwargs["top_p"] = r.sampling["top_p"]
                else:
                    gen_kwargs["do_sample"] = False

                with torch.no_grad():
                    outputs = self.llm.generate(**inputs, **gen_kwargs)
                generated_tokens = outputs[0][input_len:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if not text.strip().startswith("{"):
                    text = "{" + text
                self.usage.update(calls=1, prompt_tokens=input_len, completion_tokens=len(generated_tokens))
                results.append(text)
            except Exception as e:
                results.append(e)
        return results


Backend = VLLM | OpenRouter | HuggingFace | TransformersLocal


def load_backend(spec: str, **kwargs) -> Backend:
    """Build a backend from 'vllm:<model>', 'openrouter:<model>', 'hf:<model>', or 'hf-local:<model>'."""
    kind, sep, model = spec.partition(":")
    if not sep or kind not in ("vllm", "openrouter", "hf", "huggingface", "hf-local", "transformers", "local"):
        raise ValueError(f"model spec must be 'vllm:<model>', 'openrouter:<model>', 'hf:<model>', or 'hf-local:<model>', got {spec!r}")
    if kind == "vllm":
        return VLLM(model, **kwargs)
    elif kind in ("hf-local", "transformers", "local"):
        return TransformersLocal(model, **kwargs)
    elif kind in ("hf", "huggingface"):
        return HuggingFace(model, **kwargs)
    else:
        return OpenRouter(model, **kwargs)


def generate(backend: Backend, requests: list[Request], attempts: int = 3) -> list[BaseModel | GenerationError]:
    """Run requests through a backend, retrying (with a new seed) the ones that fail validation."""
    results: list[Any] = [None] * len(requests)
    todo = list(range(len(requests)))
    for attempt in range(attempts):
        batch = [requests[i] if requests[i].seed is None else replace(requests[i], seed=requests[i].seed + attempt) for i in todo]
        retry = []
        for i, text in zip(todo, backend.complete(batch)):
            try:
                if isinstance(text, Exception):
                    raise text
                raw = _json_span(text)
                out_schema = requests[i].output_type.model_json_schema()
                props = out_schema.get("properties", {})
                import json
                try:
                    d = json.loads(raw)
                    if isinstance(d, dict):
                        if props.get("understood", {}).get("type") == "null" and d.get("understood") is not None:
                            d["understood"] = None
                        if "triage" in props and d.get("triage") not in (1, 2, 3, 4, 5):
                            d["triage"] = 3
                        if d.get("action") == "check_vital" and not d.get("vital"):
                            vitals_prop = props.get("vital", {})
                            allowed_vitals = vitals_prop.get("enum", [])
                            if not allowed_vitals and "anyOf" in vitals_prop:
                                for opt in vitals_prop["anyOf"]:
                                    if "enum" in opt:
                                        allowed_vitals.extend(opt["enum"])
                            text_lower = ((d.get("utterance") or "") + " " + (d.get("explanation") or "")).lower()
                            matched = None
                            for v in allowed_vitals:
                                if v in text_lower or (v == "heartrate" and "pulse" in text_lower) or (v == "sbp" and "pressure" in text_lower) or (v == "o2sat" and "oxygen" in text_lower):
                                    matched = v
                                    break
                            d["vital"] = matched or (allowed_vitals[0] if allowed_vitals else None)
                            if not d["vital"]:
                                d["action"] = "utterance"
                        if d.get("action") == "utterance" and not (d.get("utterance") or "").strip():
                            d["utterance"] = "Could you tell me more about what you're experiencing?"
                        raw = json.dumps(d)
                except Exception:
                    pass
                results[i] = requests[i].output_type.model_validate_json(raw)
            except Exception as e:
                results[i] = GenerationError(f"{type(e).__name__}: {e}")
                retry.append(i)
        todo = retry
        if not todo:
            break
    return results
