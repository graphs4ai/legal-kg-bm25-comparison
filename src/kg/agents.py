"""LLM agent: prompt rendering, provider dispatch, and structured output parsing.

``Agent.run`` renders the prompt, calls the model, and validates the JSON reply
against a pydantic schema, re-asking with the error on a parse failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from src.kg.config import AgentConfig
from src.kg.prompt_registry import PromptRegistry


class AgentOutputError(Exception):
    """Raised when an agent response cannot be parsed into its schema."""


def extract_json_block(text: str) -> str:
    """Extract the first JSON object from a model response.

    Handles bare JSON, fenced JSON, and JSON surrounded by prose.

    Raises:
        AgentOutputError: If no JSON object can be located.
    """

    if not text:
        raise AgentOutputError("Empty model response")

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    if _parses(candidate) or _parses(
                            repair_trailing_commas(candidate)):
                        return candidate
                    break
        start = text.find("{", start + 1)

    raise AgentOutputError("No JSON object found in model response")


def _parses(text: str) -> bool:
    """Whether ``text`` is valid JSON, without raising."""

    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def repair_trailing_commas(text: str) -> str:
    """Drop commas immediately before a closing ``}`` or ``]``, outside strings."""

    out: list[str] = []
    in_string = False
    escaped = False
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == ",":
            j = i + 1
            while j < n and text[j] in " \t\n\r":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(char)
        i += 1
    return "".join(out)


def close_truncated_json(text: str) -> str | None:
    """Rebuild a response that stopped at the token limit into valid JSON.

    Cuts back to the last finished array element and closes the open
    containers. Returns None if nothing usable survives.
    """

    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]

    stack: list[str] = []
    in_string = False
    escaped = False
    # Depth of an open array -> offset just past its last complete element.
    array_cut: dict[int, int] = {}

    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
            if char == "[":
                array_cut[len(stack) - 1] = i + 1
        elif char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if stack and stack[-1] == "]":
                array_cut[len(stack) - 1] = i + 1
        elif char == "," and stack and stack[-1] == "]":
            array_cut[len(stack) - 1] = i

    if not stack:
        return None

    # Keep the innermost open array up to its last finished element.
    depths = [
        depth for depth in array_cut
        if depth < len(stack) and stack[depth] == "]"
    ]
    if not depths:
        return None

    repaired = text[: array_cut[max(depths)]].rstrip().rstrip(",")
    return repaired + "".join(_open_containers(repaired))


def _open_containers(text: str) -> list[str]:
    """Return the closing tokens ``text`` still needs, innermost first."""

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    return list(reversed(stack))


@dataclass
class AgentResult:
    """Outcome of one agent call.

    Attributes:
        attempts (int): Model calls made (1 = no parse retry).
        usage (dict): Tokens, cost, and seconds summed over all attempts.
        call_id (str | None): Id of the record in
            ``results/kg/<kg_version>/calls/<stage>.jsonl``, when logging is on.
        truncated (bool): The accepted response hit the token limit.
    """

    output: BaseModel
    prompt: str
    raw_text: str
    attempts: int
    usage: dict = field(default_factory=dict)
    call_id: str | None = None
    truncated: bool = False


class Agent:
    """One builder or critic agent with structured JSON output."""

    def __init__(
        self,
        config: AgentConfig,
        schema: type[BaseModel],
        registry: PromptRegistry | None = None,
        llm_fn=None,
        parse_max_retries: int = 1,
        call_logger=None,
    ):
        """
        Args:
            registry (PromptRegistry | None): Default is ``prompts/kg/``.
            llm_fn (callable | None): ``llm_fn(prompt, config) -> dict`` with a
                ``text`` key, replacing the provider call (mocks).
            parse_max_retries (int): Re-asks allowed on parse failure.
            call_logger (CallLogger | None): Call recorder; None disables it.
        """

        self.config = config
        self.schema = schema
        self.registry = registry or PromptRegistry()
        self.llm_fn = llm_fn
        self.parse_max_retries = parse_max_retries
        self.call_logger = call_logger

    def render_prompt(self, **variables: str) -> str:
        """Render this agent's prompt template."""

        return self.registry.render(
            self.config.name, self.config.prompt_version, **variables
        )

    def prompt_sha256(self) -> str | None:
        """Return the sha256 of this agent's prompt template, or None if missing."""

        import hashlib

        version = self.config.prompt_version
        if version == "latest":
            version = self.registry.latest_version(self.config.name)
        path = self.registry.path(self.config.name, version)
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _call(self, prompt: str) -> dict:
        """Send the prompt to the configured provider; returns a dict with ``text``."""

        if self.llm_fn is not None:
            return self.llm_fn(prompt, self.config)

        if self.config.provider == "deepinfra":
            from src.utils.llm_client import call_deepinfra
            return call_deepinfra(
                prompt=prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                reasoning_effort=self.config.think,
                timeout=self.config.request_timeout,
                max_retries=self.config.request_max_retries,
            )
        if self.config.provider == "ollama":
            from src.utils.llm_client import call_ollama
            return call_ollama(
                prompt=prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                think=self.config.think,
                num_predict=self.config.max_tokens,
                format="json",
                num_ctx=self.config.num_ctx,
            )
        raise ValueError(
            f"Unknown provider {self.config.provider!r} and no llm_fn set"
        )

    def parse(self, text: str, truncated: bool = False) -> BaseModel:
        """Parse a raw response into the agent's schema.

        A ``truncated`` response is salvaged first, since otherwise
        ``extract_json_block`` would accept an inner fragment of it.

        Raises:
            AgentOutputError: On missing/invalid JSON or schema mismatch.
        """

        if truncated:
            salvaged = close_truncated_json(text)
            if salvaged is not None:
                try:
                    return self.schema.model_validate(json.loads(salvaged))
                except (json.JSONDecodeError, ValidationError):
                    pass

        block = extract_json_block(text)
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            try:
                data = json.loads(repair_trailing_commas(block))
            except json.JSONDecodeError as exc:
                salvaged = close_truncated_json(text)
                if salvaged is None:
                    raise AgentOutputError(f"Invalid JSON: {exc}") from exc
                try:
                    data = json.loads(salvaged)
                except json.JSONDecodeError:
                    raise AgentOutputError(f"Invalid JSON: {exc}") from exc
        try:
            return self.schema.model_validate(data)
        except ValidationError as exc:
            raise AgentOutputError(f"Schema validation failed: {exc}") from exc

    @property
    def stage(self) -> str:
        """Pipeline stage this agent belongs to, from its name prefix."""

        return self.config.name.split("_")[0]

    @staticmethod
    def _attempt_record(response: dict, error: str | None = None) -> dict:
        """Flatten one provider response into a call-log attempt."""

        return {
            "text": response.get("text", ""),
            "thinking": response.get("thinking"),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "elapsed_seconds": response.get("elapsed_seconds"),
            "estimated_cost": response.get("estimated_cost"),
            "finish_reason": response.get("finish_reason"),
            "parse_error": error,
        }

    def _record_call(
        self,
        prompt: str,
        attempts: list[dict],
        context: dict | None,
        variables: dict | None = None,
        output=None,
        status: str = "ok",
        error: str | None = None,
    ) -> str | None:
        """Write this call to the call log; returns its call id or None."""

        if self.call_logger is None:
            return None
        record = self.call_logger.record(
            stage=self.stage,
            agent_config=self.config,
            prompt=prompt,
            attempts=attempts,
            context=context,
            variables=variables,
            output=output,
            status=status,
            error=error,
        )
        return record["call_id"] if record else None

    def run(self, _context: dict | None = None, **variables: str) -> AgentResult:
        """Render, call, and parse, re-asking with the error on a parse failure.

        Args:
            _context (dict | None): Traceability keys logged with the call
                (e.g. ``{"shard": "cf_1988", "iteration": 1}``).
            **variables: Values for the prompt template placeholders.

        Raises:
            AgentOutputError: If parsing still fails after all retries.
        """

        prompt = self.render_prompt(**variables)
        usage = {
            "input_tokens": 0, "output_tokens": 0,
            "estimated_cost": 0.0, "elapsed_seconds": 0.0,
        }

        attempt_prompt = prompt
        attempts: list[dict] = []
        last_error: AgentOutputError | None = None
        for attempt in range(1, self.parse_max_retries + 2):
            try:
                response = self._call(attempt_prompt)
            except Exception as exc:
                self._record_call(
                    prompt, attempts, _context, variables,
                    status="error", error=f"{type(exc).__name__}: {exc}",
                )
                raise

            usage["input_tokens"] += response.get("input_tokens") or 0
            usage["output_tokens"] += response.get("output_tokens") or 0
            usage["estimated_cost"] += response.get("estimated_cost") or 0.0
            usage["elapsed_seconds"] += response.get("elapsed_seconds") or 0.0

            try:
                output = self.parse(
                    response.get("text", ""),
                    truncated=response.get("finish_reason") == "length",
                )
            except AgentOutputError as exc:
                last_error = exc
                attempts.append(self._attempt_record(response, str(exc)))
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"Sua resposta anterior foi:\n{response.get('text', '')}\n\n"
                    f"Ela é inválida ({exc}). Responda novamente APENAS com um "
                    f"objeto JSON válido no formato especificado."
                )
                continue

            attempts.append(self._attempt_record(response))
            call_id = self._record_call(
                prompt, attempts, _context, variables, output=output)
            return AgentResult(
                output=output,
                prompt=prompt,
                raw_text=response.get("text", ""),
                attempts=attempt,
                usage=usage,
                call_id=call_id,
                truncated=response.get("finish_reason") == "length",
            )

        message = (
            f"Agent {self.config.name!r} failed after "
            f"{self.parse_max_retries + 1} attempts: {last_error}"
        )
        self._record_call(
            prompt, attempts, _context, variables,
            status="parse_failed", error=message)
        raise AgentOutputError(message)
