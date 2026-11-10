"""The LoRA judge: a small local model (Ollama) fine-tuned on FEVER as claim + evidence -> label.

The SAME prompt format is used to build the training data (factcheck/lora_data.py) and at
inference time - that is what makes the fine-tune work. Change it here, retrain.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentcore import llm

from .schemas import Label

SYSTEM = ("You are a fact-checking judge. Given a claim and evidence sentences, answer with JSON: "
          '{"label": "TRUE" | "FALSE" | "NOT_ENOUGH_INFO"}. Use only the evidence.')


class LoraVerdict(BaseModel):
    label: Label


def prompt(claim: str, evidence: list[str]) -> str:
    ev = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(evidence)) or "(none)"
    return f"Claim: {claim}\nEvidence:\n{ev}"


def target(label: str) -> str:
    return LoraVerdict(label=label).model_dump_json()


def judge(claim: str, evidence: list[str], model: str, backend: str = "ollama") -> Label:
    out = llm.generate(prompt(claim, evidence), schema=LoraVerdict, system=SYSTEM, backend=backend,
                       model=model, agent="judge.lora", max_output_tokens=32)
    return out.label
