import json
import logging
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class _PerplexityCalculator:
    def __init__(self, model_name: str = "xlm-roberta-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "gpt" in model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_causal = True
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.is_causal = False

        self.model = self.model.to(device)
        self.model.eval()

    def calculate_perplexity(self, text: str) -> float:
        if self.is_causal:
            return self._compute_causal_ppl(text)
        return self._compute_masked_ppl(text)

    def _compute_causal_ppl(self, text: str) -> float:
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids
        stride = 512

        losses = []
        for i in range(0, input_ids.size(1) - 1, stride):
            begin_loc = max(i - stride, 0)
            end_loc = min(i + stride, input_ids.size(1))
            input_ids_window = input_ids[:, begin_loc:end_loc]

            with torch.no_grad():
                outputs = self.model(input_ids_window, labels=input_ids_window)
                losses.append(outputs.loss.item())

        return float(torch.exp(torch.tensor(losses).mean()).item())

    def _compute_masked_ppl(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = enc["input_ids"][0]

        if len(input_ids) < 2:
            return float("inf")

        losses = []

        for i in range(len(input_ids)):
            masked = input_ids.clone()
            masked[i] = self.tokenizer.mask_token_id

            input_tensor = masked.unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits

                probs = torch.log_softmax(logits, dim=-1)

                loss = -probs[0, i, input_ids[i]].item()
                losses.append(loss)

        return float(np.exp(np.mean(losses)))


class VietnamesePerplexityEvaluator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_map = {
            "xlm-roberta": "xlm-roberta-base",
            "multilingual": "distilgpt2",
        }
        self.models = {
            label: _PerplexityCalculator(model_name=name, device=device)
            for label, name in self.model_map.items()
        }

    def evaluate_summary(self, summary: str) -> Dict[str, Union[str, int, Dict[str, float]]]:
        """Evaluate a single Vietnamese summary and return both model scores."""
        model_scores: Dict[str, float] = {}
        for label, calculator in self.models.items():
            model_scores[label] = calculator.calculate_perplexity(summary)

        return {
            "text_length": len(summary),
            "word_count": len(summary.split()),
            "model_scores": model_scores,
            "average_ppl": float(np.mean(list(model_scores.values()))),
        }

    def batch_evaluate_summaries(self, summaries: List[str]) -> Dict[str, Union[List[Dict], Dict[str, float]]]:
        """Evaluate multiple Vietnamese summaries and return per-summary model scores."""
        individual_scores = [self.evaluate_summary(summary) for summary in summaries]

        model_names = list(self.models.keys())
        model_metrics = {
            label: [item["model_scores"][label] for item in individual_scores]
            for label in model_names
        }

        return {
            "individual_scores": individual_scores,
            "statistics": {
                "mean_ppl": float(np.mean([item["average_ppl"] for item in individual_scores])),
                "std_ppl": float(np.std([item["average_ppl"] for item in individual_scores])),
                "min_ppl": float(np.min([item["average_ppl"] for item in individual_scores])),
                "max_ppl": float(np.max([item["average_ppl"] for item in individual_scores])),
                "per_model_mean": {label: float(np.mean(scores)) for label, scores in model_metrics.items()},
                "per_model_std": {label: float(np.std(scores)) for label, scores in model_metrics.items()},
                "total_summaries": len(summaries),
            },
        }


# Examples


def example_evaluate_summary() -> None:
    evaluator = VietnamesePerplexityEvaluator()
    summary = (
        "Năm nay bạn có nhiều cơ hội phát triển sự nghiệp. "
        "Hãy cân bằng công việc và tình cảm để tránh áp lực."
    )
    result = evaluator.evaluate_summary(summary)
    print("Evaluate single summary:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def example_batch_evaluate_summaries() -> None:
    evaluator = VietnamesePerplexityEvaluator()
    summaries = [
        "Bạn có năng lực lãnh đạo mạnh mẽ. Năm nay công việc sẽ thuận lợi.",
        "Tình yêu là trọng tâm. Hãy mở lòng để nhận tình cảm từ xung quanh.",
    ]
    result = evaluator.batch_evaluate_summaries(summaries)
    print("Evaluate batch summaries:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    example_evaluate_summary()
    example_batch_evaluate_summaries()

