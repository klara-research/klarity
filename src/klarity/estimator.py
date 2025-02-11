# estimator.py
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from together import Together
from transformers import PreTrainedTokenizer, LogitsProcessor
from .models import TokenInfo, UncertaintyMetrics, UncertaintyAnalysisResult
from .core.analyzer import EntropyAnalyzer, VLMAnalyzer
import math


class UncertaintyLogitsProcessor(LogitsProcessor):
    def __init__(self, estimator):
        self.captured_logits = []
        self.estimator = estimator

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.captured_logits.append(scores.detach().clone())
        return scores


class UncertaintyEstimator:
    def __init__(
        self,
        top_k: int = 100,
        analyzer: Optional[EntropyAnalyzer] = None,
        together_api_key: Optional[str] = None,
        together_model: Optional[str] = None,
    ):
        self.analyzer = analyzer
        # max log probs provided by the api
        self.top_k = 5 if together_model else top_k
        self.together_client = Together(api_key=together_api_key) if together_api_key else None
        self.together_model = together_model

    def get_logits_processor(self) -> LogitsProcessor:
        """Get appropriate logits processor based on model type"""
        return UncertaintyLogitsProcessor(self)

    def _process_logits(self, logits: torch.Tensor, tokenizer: PreTrainedTokenizer) -> List[TokenInfo]:
        """Process HuggingFace logits into TokenInfo objects"""
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], self.top_k)
        top_logits = logits[0][top_indices]

        token_info = []
        for token_id, logit, prob in zip(top_indices, top_logits, top_probs):
            token = tokenizer.decode(token_id)
            token_info.append(
                TokenInfo(
                    token=token,
                    token_id=token_id.item(),
                    logit=logit.item(),
                    probability=prob.item(),
                )
            )
        return token_info

    def _process_together_logprob(self, logprob: float) -> float:
        """Convert logprob to probability correctly"""
        return float(np.exp(logprob))

    def _generate_with_together(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using Together AI chat with logprobs"""
        messages = [{"role": "user", "content": prompt}]
        generation_config = {
            "max_tokens": kwargs.get("max_new_tokens", 10),
            "temperature": kwargs.get("temperature", 0.7),
            "logprobs": True,  # Enable logprobs for confidence tracking
            "stream": False,
        }

        response = self.together_client.chat.completions.create(
            model=self.together_model, messages=messages, **generation_config
        )

        logprobs_data = response.choices[0].logprobs
        print(logprobs_data)
        return {
            "text": response.choices[0].message.content,
            "tokens": logprobs_data.tokens,
            "token_logprobs": logprobs_data.token_logprobs,
            "token_ids": logprobs_data.token_ids,
        }

    def analyze_generation(
        self,
        generation_output: Any,
        model: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[LogitsProcessor] = None,
        prompt: Optional[str] = None,
    ) -> UncertaintyAnalysisResult:
        all_metrics = []
        generated_text = ""
        attention_data = None
        input_query = prompt or ""

        # Check if this is a VLM output
        is_vlm = hasattr(generation_output, "attentions") and isinstance(self.analyzer, VLMAnalyzer)

        if is_vlm:
            if not hasattr(self.analyzer, "patch_size") or self.analyzer.patch_size is None:
                self.analyzer.set_vision_config(model.config.vision_config)
            # Process VLM-specific outputs
            input_length = processor.input_ids.shape[1] if hasattr(processor, "input_ids") else 0
            generated_tokens = generation_output.sequences[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Get individual tokens
            tokens = []
            for i in range(len(generated_tokens)):
                token_text = tokenizer.decode(generated_tokens[i : i + 1], skip_special_tokens=True)
                if token_text.strip():
                    tokens.append(token_text)

            # Process attention maps
            attention_data = self.analyzer.process_attention_maps(generation_output.attentions, tokens)

            # Process token predictions as before
            if hasattr(generation_output, "scores"):
                for step, logits in enumerate(generation_output.scores):
                    token_info = self._process_logits(logits, tokenizer)

                    metrics = UncertaintyMetrics(
                        raw_entropy=self.analyzer._calculate_raw_entropy(np.array([t.probability for t in token_info])),
                        semantic_entropy=self.analyzer._calculate_semantic_entropy(token_info),
                        token_predictions=token_info,
                    )
                    all_metrics.append(metrics)

        else:
            # Handle VLLM outputs
            if hasattr(generation_output, "outputs"):
                generated_text = generation_output.outputs[0].text
                logprobs_data = generation_output.outputs[0].logprobs

                if logprobs_data:
                    for token_data in logprobs_data:
                        logprobs_items = [
                            (token, logprob.logprob, logprob.decoded_token) for token, logprob in token_data.items()
                        ]
                        logprobs_items.sort(key=lambda x: x[1], reverse=True)

                        token_info = [
                            TokenInfo(
                                token=decoded_token,
                                token_id=int(token_id),
                                logit=logprob,
                                probability=math.exp(logprob),
                            )
                            for token_id, logprob, decoded_token in logprobs_items[: self.top_k]
                        ]

                        if token_info:
                            probs = np.array([t.probability for t in token_info])
                            raw_entropy = (
                                self.analyzer._calculate_raw_entropy(probs)
                                if self.analyzer
                                else -np.sum(probs * np.log(probs))
                            )
                            semantic_entropy = (
                                self.analyzer._calculate_semantic_entropy(token_info) if self.analyzer else 0.0
                            )

                            metrics = UncertaintyMetrics(
                                raw_entropy=raw_entropy,
                                semantic_entropy=semantic_entropy,
                                token_predictions=token_info,
                            )
                            all_metrics.append(metrics)

            # Handle Together API outputs
            elif self.together_model:
                generated_text = generation_output["text"]
                tokens = generation_output["tokens"]
                token_logprobs = generation_output["token_logprobs"]
                token_ids = generation_output["token_ids"]

                for step in range(len(tokens)):
                    prob = self._process_together_logprob(token_logprobs[step])
                    uncertainty_score = 1 - prob

                    token_info = [
                        TokenInfo(
                            token=tokens[step],
                            token_id=token_ids[step],
                            logit=token_logprobs[step],
                            probability=prob,
                        )
                    ]

                    metrics = UncertaintyMetrics(
                        raw_entropy=uncertainty_score,
                        semantic_entropy=0.0,
                        token_predictions=token_info,
                    )
                    all_metrics.append(metrics)

            # Handle HuggingFace outputs
            else:
                if not tokenizer or not processor:
                    raise ValueError("Tokenizer and processor required for HuggingFace models")

                generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)

                for step, logits in enumerate(processor.captured_logits):
                    token_info = self._process_logits(logits, tokenizer)

                    metrics = UncertaintyMetrics(
                        raw_entropy=self.analyzer._calculate_raw_entropy(np.array([t.probability for t in token_info])),
                        semantic_entropy=self.analyzer._calculate_semantic_entropy(token_info),
                        token_predictions=token_info,
                    )
                    all_metrics.append(metrics)

        # Generate comprehensive insight based on model type
        if is_vlm:
            overall_insight = self.analyzer.generate_overall_insight(
                all_metrics, input_query=input_query, generated_text=generated_text, attention_data=attention_data
            )
        else:
            overall_insight = self.analyzer.generate_overall_insight(
                all_metrics, input_query=input_query, generated_text=generated_text
            )

        return UncertaintyAnalysisResult(
            token_metrics=all_metrics,
            overall_insight=overall_insight,
            attention_data=attention_data if is_vlm else None,
        )
