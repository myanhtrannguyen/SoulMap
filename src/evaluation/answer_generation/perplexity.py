import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class PerplexityCalculator:

    def __init__(
        self,
        model_name="bigscience/bloom-560m",
        device="cpu"
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to(device)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate(self, text: str) -> float:

        if not text or not text.strip():
            return float("inf")

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():

            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=enc["input_ids"]
            )

            loss = outputs.loss

        return float(torch.exp(loss).item())

class VietnamesePerplexityEvaluator:
    def __init__(self, device="cpu"):
        self.calculators = {
            "gpt2": PerplexityCalculator("gpt2", device=device),
        }

    def evaluate_summary(self, text: str):
        scores = {
            name: calc.calculate(text)
            for name, calc in self.calculators.items()
        }

        return {
            "text_length": len(text),
            "word_count": len(text.split()),
            "model_scores": scores,
            "average_ppl": float(np.mean(list(scores.values())))
        }

    def batch_evaluate(self, texts):
        results = [self.evaluate_summary(t) for t in texts]

        return {
            "individual_scores": results,
            "statistics": {
                "mean_ppl": float(np.mean([r["average_ppl"] for r in results])),
                "std_ppl": float(np.std([r["average_ppl"] for r in results])),
                "min_ppl": float(np.min([r["average_ppl"] for r in results])),
                "max_ppl": float(np.max([r["average_ppl"] for r in results])),
                "total": len(texts)
            }
        }