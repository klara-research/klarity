from datasets import load_dataset
from enum import Enum
from loguru import logger
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

import argparse
import numpy as np

from klarity.core.analyzer import EntropyAnalyzer

logger.add("logs/reasoning_budget_control.log")


class DatasetType(str, Enum):
    AIME = "aime"
    MATH = "math"


class BudgetMode(str, Enum):
    WAIT = "wait"
    ENTROPY = "entropy"


class VLLMClient:
    WAIT_STR: str = " Wait, "
    THINK_START_STR = "<think>"
    THINK_END_STR = "</think>"
    BOXED_START_STR = " \\boxed{"
    BOXED_END_STR = "}"

    def __init__(
        self,
        model: str = "agentica-org/DeepScaleR-1.5B-Preview",
        tensor_parallel_size: int = 2,
    ):
        self.model = LLM(
            model,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            dtype="bfloat16",
        )

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
        self.entropy_analyzer = EntropyAnalyzer()

    def add_system_prompt(self, prompt: str):
        """
        Add system prompt to the prompt.

        Args:
            prompt (str): The prompt to add the system prompt to.

        Returns:
            str: The prompt with the system prompt added.
        """
        prompt = (
            "Please reason step by step, and put your final answer within \\boxed\{\}."
            "Solve the following problem:" + prompt
        )
        return prompt

    def tokenize_prompt(self, prompt: str) -> str:
        """
        Tokenize the prompt using the tokenizer.

        Args:
            prompt (str): The prompt to tokenize.

        Returns:
            str: The text of the prompt after applying the chat template.
        """
        prompt = self.add_system_prompt(prompt)
        tokens_prompt = TokensPrompt(
            prompt_token_ids=self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
        )

        return (
            self.tokenizer.decode(tokens_prompt["prompt_token_ids"], skip_special_tokens=False) + self.THINK_START_STR
        )

    def extract_answer(self, text: str):
        """
        Extract the answer between boxed{ and }.

        Args:
            text (str): The text to extract the answer from.

        Returns:
            str: The answer extracted from the text.
        """
        try:
            last_occurrence = text.rindex("boxed{")
            end_pos = text.find("}", last_occurrence)
            if end_pos != -1:
                return text[last_occurrence + len("boxed{") : end_pos]
            return ""
        except ValueError:
            logger.error(f"Failed to parse {text[-1000:]}")
            return ""

    def get_vllm_output(self, vllm_response: List, get_logprobs: bool = False):
        """
        Get the output of the model.

        Args:
            output_generation (OutputGeneration): The output generation object.

        Returns:
            Tuple[List[str], List[int]]: The generated text, the number of generated tokens.
        """
        generated_texts = []
        num_generated_tokens = []
        logprobs = []
        token_ids = []

        for prompt_ind in range(len(vllm_response)):
            for sample_ind in range(len(vllm_response[prompt_ind].outputs)):
                generated_texts.append(vllm_response[prompt_ind].outputs[sample_ind].text)
                token_ids.append(vllm_response[prompt_ind].outputs[sample_ind].token_ids)
                num_generated_tokens.append(len(vllm_response[prompt_ind].outputs[sample_ind].token_ids))
                if get_logprobs:
                    lgps = []
                    for logprob in vllm_response[prompt_ind].outputs[sample_ind].logprobs:
                        lgp = [lg.logprob for lg in logprob.values()]
                        lgps.append(lgp)
                    logprobs.append(lgps)

        return generated_texts, num_generated_tokens, logprobs, token_ids

    def find_first_or_last_subsequence(self, arr: List[int], sub: List[int], is_first=False) -> int:
        """
        Find the index of a subsequence in a list.

        Args:
            arr (List[int]): The list to search.
            sub (List[int]): The subsequence to search for.

        Returns:
            int: The index of the occurrence of the subsequence, or -1 if not found.
        """
        if not arr or not sub:
            return -1

        step = 1 if is_first else -1
        start = 0 if is_first else len(arr) - len(sub)
        end = len(arr) - len(sub) + 1 if is_first else -1

        for i in range(start, end, step):
            if arr[i : i + len(sub)] == sub:
                return i
        return -1

    def compute_entropy_of_answer(
        self,
        response_token_ids: tuple[int],
        response_logprobs: np.array,
    ) -> float:
        """
        Finds the token indices corresponding to the most recent \\boxed{ and },
        uses the log probs of those tokens to compute entropy

        Args:
            response_token_ids: The token ids of the response.
            response_logprobs: The log probs of the response.

        Returns:
            float: Computes the mean entropy of all the tokens between \\boxed{ and }
        """

        # We need to skip the start_of_sequence special token
        boxed_tokens = tuple(self.tokenizer(self.BOXED_START_STR)["input_ids"][1:])
        boxed_end_tokens = tuple(self.tokenizer(self.BOXED_END_STR)["input_ids"][1:])

        # Find the most recent boxed tokens sequence in response
        last_boxed_start_index = self.find_first_or_last_subsequence(response_token_ids, boxed_tokens, is_first=False)

        last_boxed_end_index = self.find_first_or_last_subsequence(
            response_token_ids[last_boxed_start_index:], boxed_end_tokens, is_first=True
        )

        if last_boxed_start_index == -1:
            return 0.0

        # Compute entropy of the answer
        all_entropies = []
        for token_ind in range(
            last_boxed_start_index + len(boxed_tokens), last_boxed_start_index + last_boxed_end_index
        ):
            # Convert probs to logprobs
            step_probs = np.exp(response_logprobs[token_ind])
            entropy = self.entropy_analyzer._calculate_raw_entropy(step_probs)
            all_entropies.append(entropy)

        return np.mean(all_entropies)

    def query_with_entropy(
        self,
        prompts: List[str],
        sampling_params: SamplingParams = SamplingParams(),
        entropy_threshold: float = 0.15,
        max_entropy_iterations: int = 2,
    ):
        num_samples = sampling_params.n
        num_generated_tokens = np.zeros((len(prompts), num_samples), dtype=int)
        predicted_answers = [["" for _ in range(num_samples)] for _ in range(len(prompts))]
        predicted_logprobs = []
        generated_token_ids = []
        max_iterations = max_entropy_iterations

        if max_iterations == 0:
            sampling_params.stop = None
        else:
            sampling_params.stop = [self.THINK_END_STR]

        for ind, prompt in enumerate(prompts):
            vllm_output = self.model.generate([prompt], sampling_params)
            generated_texts, num_tokens, logprobs, token_ids = self.get_vllm_output(vllm_output, get_logprobs=True)
            num_generated_tokens[ind] = num_tokens
            predicted_answers[ind] = generated_texts
            generated_token_ids.append(token_ids)
            predicted_logprobs.append(logprobs)
            for sample_ind in range(num_samples):
                entire_chat_history = predicted_answers[ind][sample_ind].rstrip(self.THINK_END_STR) + self.WAIT_STR
                sampling_params.n = 1
                is_reasoning_complete = False
                for iteration in range(max_entropy_iterations):
                    entropy = self.compute_entropy_of_answer(
                        generated_token_ids[ind][sample_ind], predicted_logprobs[ind][sample_ind]
                    )

                    if entropy < entropy_threshold or max_entropy_iterations-1 == iteration:
                        sampling_params.stop = None
                        is_reasoning_complete = True
                        entire_chat_history = entire_chat_history.rstrip(self.WAIT_STR) + self.THINK_END_STR

                    vllm_output = self.model.generate(entire_chat_history, sampling_params)
                    generated_texts, num_tokens, logprobs, token_ids = self.get_vllm_output(vllm_output, get_logprobs=True)
                    predicted_logprobs[ind][sample_ind] = logprobs
                    num_generated_tokens[ind][sample_ind] += num_tokens[0]
                    predicted_answers[ind][sample_ind] = entire_chat_history + generated_texts[0]
                    generated_token_ids[ind][sample_ind] = token_ids

                    if is_reasoning_complete:
                        sampling_params.stop = [self.THINK_END_STR]
                        sampling_params.n = num_samples
                        break

        # Get total number of tokens generated per sampled response i.e sum along wait axis
        predicted_answers = [
            [
                self.extract_answer(predicted_answers[prompt_ind][sample_ind])
                for sample_ind in range(len(predicted_answers[prompt_ind]))
            ]
            for prompt_ind in range(len(predicted_answers))
        ]
        return num_generated_tokens, predicted_answers

    def query_with_extra_wait(
        self,
        prompts: List[str],
        num_waits: int = 1,
        sampling_params: SamplingParams = SamplingParams(),
    ):
        """
        Query the model with the given prompt.

        Args:
        - prompts (List[str]): The prompt to generate text for.
        - num_waits (int, optional): The number of extra waits to add. Defaults to 1.
        - sampling_params (SamplingParams, optional): The sampling parameters to use. Defaults to SamplingParams().

        Returns:
        - str: The generated text and the number of tokens generated.
        """

        if num_waits < 0:
            raise ValueError("num_waits must be non-negative")
        elif num_waits == 0:
            stop = None
        else:
            stop = [self.THINK_END_STR]

        sampling_params.stop = stop
        num_samples = sampling_params.n
        # Tokenize prompt
        num_generated_tokens = np.zeros((len(prompts), num_samples), dtype=int)
        predicted_answers = [["" for _ in range(num_samples)] for _ in range(len(prompts))]
        entire_chat_history = ""

        for ind, prompt in enumerate(prompts):
            sampling_params.n = num_samples
            input_text = [prompt]
            # Get first n reasoning trajectories
            outputs = self.model.generate(input_text, sampling_params=sampling_params)
            generated_texts, token_counts, _, _ = self.get_vllm_output(outputs)
            num_generated_tokens[ind] += token_counts
            predicted_answers[ind] = generated_texts
            for sample_ind in range(num_samples):
                entire_chat_history = predicted_answers[ind][sample_ind].rstrip(self.THINK_END_STR) + self.WAIT_STR
                sampling_params.n = 1
                for wait_ind in range(num_waits):
                    if wait_ind == num_waits - 1:
                        # If this is the last wait, set stop to None to ensure full generation
                        sampling_params.stop = None
                    try:
                        # Generate text using the model
                        outputs = self.model.generate(entire_chat_history, sampling_params=sampling_params)

                        # Process model outputs
                        generated_texts, token_counts, _, _ = self.get_vllm_output(outputs)
                        num_generated_tokens[ind][sample_ind] += token_counts[0]

                        # TODO: only consider the last generation
                        predicted_answers[ind][sample_ind] = entire_chat_history + generated_texts[0]
                        entire_chat_history += generated_texts[0].rstrip(self.THINK_END_STR) + self.WAIT_STR

                        logger.info(f"Prefix for wait: {wait_ind} of input is {entire_chat_history[:500]}")
                        logger.info(f"Suffix for wait: {wait_ind} of input is {entire_chat_history[-200:]}")

                    except Exception as e:
                        # Log the error and handle gracefully
                        logger.error(f"Error during text generation: {str(e)}")
                        predicted_answers[ind][sample_ind] = ""
                        num_generated_tokens[ind][sample_ind] = -float("inf")

        # Get total number of tokens generated per sampled response i.e sum along wait axis
        predicted_answers = [
            [
                self.extract_answer(predicted_answers[prompt_ind][sample_ind])
                for sample_ind in range(len(predicted_answers[prompt_ind]))
            ]
            for prompt_ind in range(len(predicted_answers))
        ]
        return num_generated_tokens, predicted_answers

    def load_aime_dataset(self, dataset_path: str = "HuggingFaceH4/aime_2024") -> List[Dict[str, str]]:
        """
        Load the aime dataset.

        Args:
        - dataset_path (str): The path to the dataset.

        Returns:
        - List[Dict[str, str]]: The dataset.
        """
        dataset = load_dataset(dataset_path)
        dataset = dataset["train"]
        final_dataset = []
        for row in dataset:
            final_dataset.append({"query": row["problem"], "answer": row["answer"]})

        return final_dataset

    def load_math_dataset(self, dataset_path: str = "HuggingFaceH4/MATH-500") -> List[Dict[str, str]]:
        """
        Load the math dataset.

        Args:
        - dataset_path (str): The path to the dataset.

        Returns:
        - List[Dict[str, str]]: The dataset.
        """
        dataset = load_dataset(dataset_path)
        dataset = dataset["test"]
        final_dataset = []
        for row in dataset:
            final_dataset.append({"query": row["problem"], "answer": row["answer"]})

        return final_dataset

    def compute_math_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Compute the accuracy of the predictions compared to the ground truth for math problems.

        Args:
        - predictions (List[str]): The predicted answers.
        - ground_truths (List[str]): The ground truth answers.

        Returns:
        - float: The accuracy of the predictions.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        if len(predictions) == 0:
            return 0.0

        correct = 0
        total = len(predictions)
        for pred, gt in zip(predictions, ground_truths):
            try:
                equal = float(pred) == float(gt)
                if equal:
                    correct += 1
            except ValueError:
                pass
        return correct / total

    def plot_benchmarks(
        self,
        accuracy_across_samples: List[float],
        average_tokens_across_samples: List[float],
        mode: BudgetMode = BudgetMode.WAIT,
        dataset_type: DatasetType = DatasetType.AIME,
    ):
        """
        Plots two subplots of how the accuracy and average tokens change across samples.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        sample_indices = range(len(accuracy_across_samples))

        ax1.plot(sample_indices, accuracy_across_samples)
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy Across Samples")

        ax2.plot(sample_indices, average_tokens_across_samples)
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Average Tokens")
        ax2.set_title("Average Tokens Across Samples")

        plt.title("Benchmark Results - " + dataset_type.value + "using " + mode.value + " for budgeting.")
        plt.tight_layout(pad=3.0)
        fig.savefig(f"plots/{mode.value}_{dataset_type.value}_accuracy_vs_tokens.png")

    def main(
        self,
        num_waits: int = 0,
        num_sample_responses: int = 1,
        dataset_type: DatasetType = DatasetType.AIME,
        max_tokens: int = 16384,
        temperature: float = 0.6,
        top_p: float = 0.95,
        budget_mode: BudgetMode = BudgetMode.WAIT,
        entropy_threshold: Optional[float] = 0.5,
        top_k: Optional[int] = 5,
        max_entropy_iterations: int = 2,
    ):
        """
        Main function to run the benchmark.

        Args:
        - dataset_type (DatasetType, optional): The type of dataset to use. Defaults to DatasetType.AIME.
        - num_waits (int, optional): The number of extra waits to add. Defaults to 0.
        - max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 16384.
        - temperature (float, optional): The temperature to use. Defaults to 0.6.
        - top_p (float, optional): The top-p to use. Defaults to 0.95.
        - budget_mode (BudgetMode, optional): The budget mode to use. Defaults to BudgetMode.WAIT.
        - entropy_threshold (Optional[float], optional): The entropy threshold to use. Defaults to 0.5.
        - top_k (Optional[int], optional): The top-k tokens considered for calculating entropy. Defaults to 5.
        - max_entropy_iterations (int, optional):
            The maximum number of iterations for entropy calculation. Defaults to 2.

        Returns:
        - None
        """
        if dataset_type == DatasetType.AIME.value:
            dataset = vllm_client.load_aime_dataset()
        elif dataset_type == DatasetType.MATH.value:
            dataset = vllm_client.load_math_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        gt_answers = []
        queries = []

        for row in tqdm(dataset, desc="Dataset row", total=len(dataset)):
            queries.append(row["query"])
            gt_answers.append(row["answer"])

        inputs = [self.tokenize_prompt(query) for query in queries]

        if budget_mode == BudgetMode.WAIT:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_sample_responses,
                skip_special_tokens=False,
            )
            total_generated_tokens, predicted_answers = self.query_with_extra_wait(inputs, num_waits, sampling_params)
        elif budget_mode == BudgetMode.ENTROPY:
            sampling_params = SamplingParams(
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=num_sample_responses, logprobs=top_k
            )
            total_generated_tokens, predicted_answers = self.query_with_entropy(
                inputs, sampling_params, entropy_threshold, max_entropy_iterations
            )
        else:
            raise ValueError(f"Unknown budget mode: {budget_mode}")

        self.compute_metrics(total_generated_tokens, predicted_answers, gt_answers, budget_mode, dataset_type)

    def compute_metrics(
        self,
        total_generated_tokens: np.array,
        predicted_answers: List[List[str]],
        gt_answers: List[str],
        mode: BudgetMode = BudgetMode.WAIT,
        dataset_type: DatasetType = DatasetType.AIME,
    ) -> Tuple[float, float, float]:
        """
        Computes metrics for the given data and plot how accuracy
            and average number of tokens change across samples.
        Args:
            total_generated_tokens (np.array): The total number of tokens generated.
            predicted_answers (List[List[str]]): The predicted answers.
            gt_answers (List[str]): The ground truth answers.
        Returns:
            Tuple[float, float, float]: The accuracy for each sample index,
                the average number of tokens and pass@1 score.
        """
        if  len(total_generated_tokens) == 0 or  len(predicted_answers) == 0 or len(gt_answers) == 0:
            raise ValueError("All inputs must not be None")

        # Shape (dataset_len, num_sample_responses)
        total_generated_tokens = np.array(total_generated_tokens)
        predicted_answers = np.array(predicted_answers)
        gt_answers = np.array(gt_answers)

        # Average number of tokens per sample index
        average_tokens_across_samples = np.mean(total_generated_tokens, axis=0)
        logger.debug(f"Average number of tokens generated: {average_tokens_across_samples}")

        # Compute accuracy across sample index
        accuracy_per_sample_ind = [
            self.compute_math_accuracy(predicted_answers[:, i], gt_answers) for i in range(len(predicted_answers[0]))
        ]
        logger.debug(f"Accuracy per sample index: {accuracy_per_sample_ind}")

        pass_1 = np.mean(accuracy_per_sample_ind) * 100
        logger.debug(f"Pass@1 for {len(predicted_answers[0])}: {pass_1}%")

        self.plot_benchmarks(accuracy_per_sample_ind, average_tokens_across_samples, mode, dataset_type)
        return accuracy_per_sample_ind, average_tokens_across_samples, pass_1


if __name__ == "__main__":
    # Accept args from the user
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_waits", type=int, default=1, help="Number of extra waits to add")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens for the query")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for the query")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for the query")
    parser.add_argument(
        "--dataset_type",
        type=DatasetType,
        default=DatasetType.AIME.value,
        help="Dataset type has to be either aime or math",
        choices=[DatasetType.AIME.value, DatasetType.MATH.value],
    )
    parser.add_argument("--num_sample_responses", type=int, default=4, help="Number of sample responses to generate")
    parser.add_argument(
        "--budget_mode",
        type=BudgetMode,
        default=BudgetMode.ENTROPY.value,
        help="Budget mode has to be either wait or entropy",
        choices=[BudgetMode.WAIT.value, BudgetMode.ENTROPY.value],
    )
    parser.add_argument(
        "--entropy_threshold", type=float, default=0.5, help="Entropy threshold to use for entropy budget control"
    )
    parser.add_argument(
        "--max_entropy_iterations", type=int, default=1, help="Maximum number of iterations for entropy calculation"
    )
    args = parser.parse_args()
    if args.dataset_type == DatasetType.AIME.value:
        dataset_type = DatasetType.AIME
    elif args.dataset_type == DatasetType.MATH.value:
        dataset_type = DatasetType.MATH

    if args.budget_mode == BudgetMode.WAIT.value:
        budget_mode = BudgetMode.WAIT
    elif args.budget_mode == BudgetMode.ENTROPY.value:
        budget_mode = BudgetMode.ENTROPY

    args = parser.parse_args()
    vllm_client = VLLMClient()

    vllm_client.main(
        num_waits=args.num_waits,
        dataset_type=dataset_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_sample_responses=args.num_sample_responses,
        budget_mode=budget_mode,
        entropy_threshold=args.entropy_threshold,
        max_entropy_iterations=args.max_entropy_iterations,
    )
