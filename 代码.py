def soft_jaccard(text_ents, img_ents, nlp, idf, thresh=0.5):
    # 预取向量
    vec = lambda e: nlp(e).vector
    def best_sim(e, others):
        ve = vec(e)
        return max(
            (float(ve @ vec(o) / (np.linalg.norm(ve)*np.linalg.norm(vec(o))) )
             for o in others),
            default=0.0)

    inter_score = 0.0
    for e in text_ents:
        s = best_sim(e, img_ents)
        if s >= thresh:
            inter_score += idf[e] * s
    for e in img_ents:
        s = best_sim(e, text_ents)
        if s >= thresh:
            inter_score += idf[e] * s

    union_weight = sum(idf[e] for e in text_ents | img_ents)
    return 0 if union_weight == 0 else inter_score / union_weight

import re, spacy, random
from typing import List, Dict, Optional
from src.model import build_model
from src.image_handler.image_scorer import IMAGE_SCORER
from src.image_handler.image_scorer.image_scorer import ImageScorer

nlp = spacy.load("en_core_web_sm")   # 💡

VLM_SCORE_PROMPT_TEMPLATE = """# Task Description
You are an assistant ...
# Scoring Strategy
...
# Reasoning Guideline (Implicit-CoT)
First, **think step-by-step**: identify entities, build their relations, reason over the graph.  
**Do NOT output your reasoning**, only keep it internal.
# Question
{question}
# Final Answer
Output only the integer score.
"""

@IMAGE_SCORER.register_module
class VLMImageScorer(ImageScorer):
  # ------------------------- constructor ------------------------- #
  def __init__(
      self,
      *,
      filter_config: Dict[str, Any] | None = None,
      enable_cache: bool = False,
      name: Optional[str] = None,
      image_url: Optional[str] = None,
      image_root: Optional[str] = None,
      model_config: Dict[str, Any] | None = None,
      use_cot: bool = True,
      use_kg: bool = True,
  ) -> None:
      if model_config is None:
          model_config = {}

      self.model = build_model(model_config)
      assert self.model.name.startswith(
          "vlm"
      ), f"a VLM is required for image scoring, but got {self.model.name}"

      self.use_cot = bool(use_cot)
      self.use_kg = bool(use_kg and _NLP is not None)

      super().__init__(
          filter_config=filter_config or {},
          enable_cache=enable_cache,
          name=self.model.name if name is None else name,
          image_url=image_url,
          image_root=image_root,
      )

  # --------------------- score-parsing helper -------------------- #
  @staticmethod
  def _get_score(text: str) -> Optional[int]:
      """
      Extract the last integer within 0-10 from the model reply.
      Returns ``None`` if nothing valid found.
      """
      matches = re.findall(r"\b(?:10|[0-9])\b", text)
      if not matches:
          return None
      score = int(matches[-1])
      return max(0, min(10, score))

  # -------------------- lightweight KG helper ------------------- #
  def _draft_kg(self, text: str, caption: str) -> str:
      """
      Build a *very* small set of triples like:
          (dog) --possible_match--> (dog)

      The purpose is only to surface overlapping entity words.
      """
      if not self.use_kg:
          return ""

      # extract nouns / proper nouns from both sources
      doc_txt = _NLP(text.lower())  # type: ignore
      doc_cap = _NLP(caption.lower())  # type: ignore

      txt_ents = {tok.text for tok in doc_txt if tok.pos_ in {"NOUN", "PROPN"}}
      cap_ents = {tok.text for tok in doc_cap if tok.pos_ in {"NOUN", "PROPN"}}

      inter = txt_ents & cap_ents
      if not inter:
          return ""

      triples = "\n".join(f"({e}) --possible_match--> ({e})" for e in sorted(inter))
      return f"\n# Knowledge Graph Context\n{triples}\n"

  # ---------------------- public API (single) -------------------- #
  def get_image_score_single(
      self, image_url: str, reference_text: Optional[str] = None, **kwargs
  ) -> Optional[Dict[str, Any]]:
      """
      Score one image–question pair.  Returns ``{"score": int, "usage": {...}}``
      or ``None`` on failure.
      """
      if reference_text is None:
          reference_text = ""

      # ------------------------------------------------------------------
      # obtain a caption if KG hints are enabled
      # (your model must implement .describe_image; otherwise fallback to "")
      # ------------------------------------------------------------------
      caption: str = ""
      if self.use_kg and hasattr(self.model, "describe_image"):
          try:
              caption = self.model.describe_image(image_url)
          except Exception:
              caption = ""

      kg_block = self._draft_kg(reference_text, caption)

      # ------------------------------------------------------------------
      # build prompt – possibly strip CoT section
      # ------------------------------------------------------------------
      if self.use_cot:
          prompt = VLM_SCORE_PROMPT_TEMPLATE
      else:
          prompt = (
              VLM_SCORE_PROMPT_TEMPLATE.split("# Reasoning Guideline")[0]
              + "\n# Question\n{question}\n\n# Final Answer\nOutput only the integer score.\n"
          )

      messages = [
          {"type": "image_url", "image_url": {"url": image_url}},
          {"type": "text", "text": prompt.format(question=reference_text) + kg_block},
      ]

      # ------------------------------------------------------------------
      # ask the VLM
      # ------------------------------------------------------------------
      response = self.model.chat(messages=messages)
      if response is None:
          return None

      return {
          "score": self._get_score(response.get("response", "")),
          "usage": response.get("usage", {}),
      }

  # ---------------- public API (batch; inward wrapper) ----------- #
  def _get_image_scores(
      self, image_urls: List[str], reference_text: Optional[str] = None
  ) -> List[Optional[Dict[str, Any]]]:
      """
      Convenience wrapper – simply calls *get_image_score_single* for each URL.
      """
      return [
          self.get_image_score_single(url, reference_text=reference_text)
          for url in image_urls
      ]

cfg = {"name": "vlm-gpt4o", "temperature": 0.0}
scorer = VLMImageScorer(model_config=cfg, use_cot=True, use_kg=True)

ret = scorer.get_image_score_single(
        image_url="https://example.com/paris.jpg",
        reference_text="Which famous Paris landmark is shown in the photo?")
print("score =", ret["score"])


import re, spacy, random
from typing import List, Dict, Optional
from src.model import build_model
from src.image_handler.image_scorer import IMAGE_SCORER
from src.image_handler.image_scorer.image_scorer import ImageScorer

nlp = spacy.load("en_core_web_sm")   # 💡

VLM_SCORE_PROMPT_TEMPLATE = """# Task Description
You are an assistant ...
# Scoring Strategy
...
# Reasoning Guideline (Implicit-CoT)
First, **think step-by-step**: identify entities, build their relations, reason over the graph.  
**Do NOT output your reasoning**, only keep it internal.
# Question
{question}
# Final Answer
Output only the integer score.
"""

@IMAGE_SCORER.register_module
class VLMImageScorer(ImageScorer):
  # ---------------- constructor ---------------- #
  def __init__(
      self,
      *,
      filter_config: Dict[str, Any] | None = None,
      enable_cache: bool = False,
      name: Optional[str] = None,
      image_url: Optional[str] = None,
      image_root: Optional[str] = None,
      model_config: Dict[str, Any] | None = None,
      use_cot: bool = True,
      use_kg: bool = True,
      soft_thresh: float = 0.5,
  ) -> None:
      if model_config is None:
          model_config = {}

      self.model = build_model(model_config)
      assert self.model.name.startswith("vlm"), (
          f"need VLM for image scoring, got {self.model.name}"
      )

      self.use_cot = bool(use_cot)
      self.use_kg = bool(use_kg)
      self.tau = float(soft_thresh)

      super().__init__(
          filter_config=filter_config or {},
          enable_cache=enable_cache,
          name=self.model.name if name is None else name,
          image_url=image_url,
          image_root=image_root,
      )

  # ------------- parse int from VLM reply --------------- #
  @staticmethod
  def _extract_last_int(text: str) -> Optional[int]:
      m = re.findall(r"\b(?:10|[0-9])\b", text)
      return int(m[-1]) if m else None

  # ------------ entity helper --------------------------- #
  @staticmethod
  def _noun_set(doc: "spacy.tokens.Doc") -> set[str]:
      return {t.text.lower() for t in doc if t.pos_ in {"NOUN", "PROPN"}}

  # ------------ soft-Jaccard & KG block ----------------- #
  def _kg_and_similarity(
      self, text: str, caption: str
  ) -> Tuple[str, int]:
      """
      • Knowledge-Graph block  
      • Jaccard → Score (0-10)
      """
      if not self.use_kg:
          return "", 0

      doc_txt = _NLP(text)
      doc_cap = _NLP(caption)

      A = self._noun_set(doc_txt)
      B = self._noun_set(doc_cap)

      if not A and not B:
          return "", 0

      # ---------- soft intersection weighted by IDF --------- #
      def best_sim(e: str, others: set[str]) -> float:
          ve = _vector(e)
          if not ve.any():
              return 0.0
          norms_e = np.linalg.norm(ve)
          best = 0.0
          for o in others:
              vo = _vector(o)
              if not vo.any():
                  continue
              sim = float(ve @ vo / (norms_e * (np.linalg.norm(vo) + 1e-8)))
              if sim > best:
                  best = sim
          return best if best >= self.tau else 0.0

      inter_score = 0.0
      for a in A:
          inter_score += _idf(a) * best_sim(a, B)
      for b in B:
          inter_score += _idf(b) * best_sim(b, A)

      union_weight = sum(_idf(e) for e in A | B)
      S_soft = inter_score / union_weight if union_weight else 0.0
      score_int = int(round(10 * S_soft))

      # ---------- KG triples (只保留硬 overlap 方便人读) ---------- #
      hard_overlap = A & B
      kg_block = ""
      if hard_overlap:
          triples = "\n".join(
              f"({e}) --possible_match--> ({e})" for e in sorted(hard_overlap)
          )
          kg_block = f"\n# Knowledge Graph Context\n{triples}\n"

      return kg_block, score_int

  # ------------------ single request -------------------- #
  def get_image_score_single(
      self,
      image_url: str,
      reference_text: Optional[str] = None,
      **kwargs,
  ) -> Optional[Dict[str, Any]]:
      question = reference_text or ""

      # (1) caption (for entity extraction)
      caption = ""
      if self.use_kg and hasattr(self.model, "describe_image"):
          try:
              caption = self.model.describe_image(image_url)
          except Exception:
              caption = ""

      kg_block, rule_score = self._kg_and_similarity(question, caption)

      # (2) build prompt
      if self.use_cot:
          prompt = VLM_SCORE_PROMPT_TEMPLATE
      else:
          prompt = (
              VLM_SCORE_PROMPT_TEMPLATE.split("# Reasoning Guideline")[0]
              + "\n# Question\n{question}\n# Final Answer\n"
          )

      messages = [
          {"type": "image_url", "image_url": {"url": image_url}},
          {"type": "text", "text": prompt.format(question=question) + kg_block},
      ]

      # (3) query VLM
      response = self.model.chat(messages=messages)

      model_score = (
          self._extract_last_int(response["response"])
          if response is not None
          else None
      )

      final_score = model_score if model_score is not None else rule_score
      return (
          {
              "score": final_score,
              "usage": response.get("usage", {}) if response else {},
          }
          if final_score is not None
          else None
      )

  # ---------------- batch wrapper ----------------------- #
  def _get_image_scores(
      self,
      image_urls: List[str],
      reference_text: Optional[str] = None,
  ) -> List[Optional[Dict[str, Any]]]:
      return [
          self.get_image_score_single(url, reference_text=reference_text)
          for url in image_urls
      ]