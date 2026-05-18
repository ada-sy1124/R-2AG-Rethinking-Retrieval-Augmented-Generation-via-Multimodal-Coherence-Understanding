from vlm_image_scorer import VLMImageScorer


if __name__ == "__main__":
    cfg = {"name": "vlm-gpt4o", "temperature": 0.0}
    scorer = VLMImageScorer(model_config=cfg, use_cot=True, use_kg=True)

    ret = scorer.get_image_score_single(
        image_url="https://example.com/paris.jpg",
        reference_text="Which famous Paris landmark is shown in the photo?",
    )
    print("score =", ret["score"] if ret else None)
