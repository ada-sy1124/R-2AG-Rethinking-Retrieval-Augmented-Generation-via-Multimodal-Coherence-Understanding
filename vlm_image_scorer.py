"""Minimal VLM image scorer (single core file)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vlm_utils import noun_set, soft_jaccard


VLM_SCORE_PROMPT_TEMPLATE = """# Task Description
You are an assistant that scores image relevance to a question from 0 to 10.
# Scoring Strategy
Score higher when entities and relations in the image align with the question.
# Reasoning Guideline (Implicit-CoT)
First, think step-by-step: identify entities, build relations, reason over the graph.
Do NOT output your reasoning; keep it internal.
# Question
{question}
# Final Answer
Output only the integer score.
"""


@dataclass
class DemoVLM:
    name: str = "vlm-demo"

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        _ = messages
        return {"response": "7", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    def describe_image(self, image_url: str) -> str:
        if "paris" in image_url.lower():
            return "a photo of the Eiffel Tower in Paris"
        return "an outdoor photo"


def build_model(model_config: Dict[str, Any]) -> DemoVLM:
    name = model_config.get("name", "vlm-demo")
    return DemoVLM(name=name)


class VLMImageScore
    def __init__(
        self,
        *,
        model_config: Optional[Dict[str, Any]] = None,
        use_cot: bool = True,
        use_kg: bool = True,
        soft_thresh: float = 0.5,
    ) -> None:
        self.model = build_model(model_config or {})
        assert self.model.name.startswith("vlm"), f"need VLM for image scoring, got {self.model.name}"
        self.use_cot = bool(use_cot)
        self.use_kg = bool(use_kg)
        self.soft_thresh = float(soft_thresh)

    @staticmethod
    def _extract_last_int(text: str) -> Optional[int]:
        m = re.findall(r"\b(?:10|[0-9])\b", text)
        return int(m[-1]) if m else None

    def _kg_and_similarity(self, text: str, caption: str) -> Tuple[str, int]:
        if not self.use_kg:
            return "", 0

        q_ents = noun_set(text)
        c_ents = noun_set(caption)
        sim = soft_jaccard(q_ents, c_ents, thresh=self.soft_thresh)
        score_int = int(round(10 * sim))

        overlap = q_ents & c_ents
        if not overlap:
            return "", score_int

        triples = "\n".join(f"({e}) --possible_match--> ({e})" for e in sorted(overlap))
        return f"\n# Knowledge Graph Context\n{triples}\n", score_int

    def get_image_score_single(
        self, image_url: str, reference_text: Optional[str] = None, **kwargs
    ) -> Optional[Dict[str, Any]]:
        question = reference_text or ""

        caption = ""
        if self.use_kg and hasattr(self.model, "describe_image"):
            try:
                caption = self.model.describe_image(image_url)
            except Exception:
                caption = ""

        kg_block, rule_score = self._kg_and_similarity(question, caption)

        if self.use_cot:
            prompt = VLM_SCORE_PROMPT_TEMPLATE
        else:
            prompt = (
                VLM_SCORE_PROMPT_TEMPLATE.split("# Reasoning Guideline")[0]
                + "\n# Question\n{question}\n\n# Final Answer\nOutput only the integer score.\n"
            )

        messages = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt.format(question=question) + kg_block},
        ]
        response = self.model.chat(messages=messages)
        if response is None:
            return None

        model_score = self._extract_last_int(response.get("response", ""))
        final_score = model_score if model_score is not None else rule_score
        return {"score": final_score, "usage": response.get("usage", {})}

    def get_image_scores(
        self, image_urls: List[str], reference_text: Optional[str] = None
    ) -> List[Optional[Dict[str, Any]]]:
        return [self.get_image_score_single(url, reference_text=reference_text) for url in image_urls]
