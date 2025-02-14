from datasets import load_dataset
from enum import Enum
from loguru import logger
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

import argparse
import numpy as np

logger.add("logs/reasoning_budget_control.log")

class DatasetType(str, Enum):
    AIME = "aime"
    MATH = "math"

class VLLMClient:

    WAIT_STR: str = " Wait, "
    THINK_START_STR = "<think>"
    THINK_END_STR = "</think>"
    def __init__(
        self, 
        model:str = "agentica-org/DeepScaleR-1.5B-Preview", 
        tensor_parallel_size: int = 1, 
        ):
        
        self.model = LLM(
            model,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            dtype="bfloat16",
        )

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
    
    def add_system_prompt(self, prompt: str):
        """
        Add system prompt to the prompt.

        Args:
            prompt (str): The prompt to add the system prompt to.

        Returns:
            str: The prompt with the system prompt added.
        """
        prompt = "Please reason step by step, and put your final answer within \\boxed\{\}."\
            "Solve the following problem:"\
            + prompt
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
            prompt_token_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        ))

        return self.tokenizer.decode(tokens_prompt["prompt_token_ids"], skip_special_tokens=False) + self.THINK_START_STR

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
                return text[last_occurrence+len("boxed{"): end_pos]
            return ""
        except ValueError:
            logger.error(f"Failed to parse {text}")
            return ""
    
    def get_vllm_output(self, output_generation):
        """
        Get the output of the model.

        Args:
            output_generation (OutputGeneration): The output generation object.

        Returns:
            Tuple[List[str], List[int]]: The generated text, the number of generated tokens.
        """
        generated_texts = []
        num_generated_tokens = []
        
        for i in range(len(output_generation.outputs)):
            generated_texts.append(output_generation.outputs[i].text)
            num_generated_tokens.append(len(output_generation.outputs[i].token_ids))
        
        return generated_texts, num_generated_tokens

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

        # Tokenize prompt
        num_generated_tokens = []
        predicted_answers = []
        for ind, prompt in enumerate(prompts):
            input_text = prompt
            num_generated_tokens.append([])
            predicted_answers.append([])
            for wait_ind in range(num_waits+1):
                if wait_ind == num_waits:
                    # If this is the last wait, set stop to None to ensure full generation
                    sampling_params.stop = None
                try:
                    # Generate text using the model
                    outputs = self.model.generate(input_text, sampling_params=sampling_params)
                    
                    # Process model outputs
                    processed_outputs = [self.get_vllm_output(output) for output in outputs][0]
                    generated_texts, token_counts = processed_outputs
                    
                    # Add wait string if needed
                    if wait_ind < num_waits:
                        input_text = [input_text + text.rstrip(self.THINK_END_STR) + self.WAIT_STR for text in generated_texts]
                    else:
                        input_text = [input_text[i] + generated_texts[i] for i in range(len(generated_texts))]
                    logger.info(f"Prefix of input is {input_text[0][:1000]}")
                    num_generated_tokens[ind].append(list(token_counts))
                    generated_texts = input_text
                    
                except Exception as e:
                    # Log the error and handle gracefully
                    logger.error(f"Error during text generation: {str(e)}")
                    batch_size = len(prompt)
                    num_generated_tokens[ind].append([-1] * batch_size)

            # We want to find the total number of tokens generated per input prompt so we need to sum
            # along the wait axis.
            predicted_answers.append([self.extract_answer(text) for text in generated_texts])

        num_generated_tokens = np.array(num_generated_tokens).sum(2)
        # Get the entire generated text and the total generated token count
        return sum(num_generated_tokens), predicted_answers
    
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
    
    def plot_benchmarks(self, accuracy_across_samples: List[float], average_tokens_across_samples: List[float]):
        """
        Plots two subplot of how the accuracy and average tokens change across samples.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.plot(accuracy_across_samples)
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Accuracy")
        
        ax2.plot(average_tokens_across_samples)
        ax2.set_xlabel("Sample")
        ax2.set_ylabel("Average Tokens")
        
        fig.savefig("plots/accuracy_vs_tokens.png")

    def main(
        self,
        num_waits: int = 0,
        num_sample_responses: int = 1,
        dataset_type: DatasetType = DatasetType.AIME,
        max_tokens: int = 16384,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ):  
        """
        Main function to run the benchmark.

        Args:
        - dataset_type (DatasetType, optional): The type of dataset to use. Defaults to DatasetType.AIME.
        - num_waits (int, optional): The number of extra waits to add. Defaults to 0.
        - max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 16384.
        - temperature (float, optional): The temperature to use. Defaults to 0.6.
        - top_p (float, optional): The top-p to use. Defaults to 0.95.

        Returns:
        - None
        """
        if dataset_type == DatasetType.AIME.value:
            dataset = vllm_client.load_aime_dataset()
        elif dataset_type == DatasetType.MATH.value:
            dataset = vllm_client.load_math_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n = num_sample_responses,
            skip_special_tokens=False
        )

        gt_answers = []
        queries = []

        for row in tqdm(dataset, desc="Dataset row", total=len(dataset)):
            queries.append(row["query"])
            gt_answers.append(row["answer"])

        inputs = [self.tokenize_prompt(query) for query in queries]
        
        total_generated_tokens, predicted_answers = self.query_with_extra_wait(
            inputs, 
            num_waits, 
            sampling_params
        )
        
        self.compute_metrics(total_generated_tokens, predicted_answers, gt_answers)

    def compute_metrics(self, total_generated_tokens, predicted_answers, gt_answers):
        """
        Computes metrics for the given data and plot how accuracy 
            and average number of tokens change across samples.
        Args:
            total_generated_tokens (List[List[int]]): The total number of tokens generated.
            predicted_answers (List[List[str]]): The predicted answers.
            gt_answers (List[List[str]]): The ground truth answers.
        Returns:
            Tuple[float, float]: The accuracy and pass@1 score.
        """
        # Shape (dataset_len, num_sample_responses)
        total_generated_tokens = np.array(total_generated_tokens)
        predicted_answers = np.array(predicted_answers)
        gt_answers = np.array(gt_answers)

        # Average number of tokens per sample index
        average_tokens_across_samples = np.mean(total_generated_tokens, axis=1)
        logger.debug(f"Average number of tokens generated: {average_tokens_across_samples}")
        
        # Compute accuracy across sample index
        accuracy_per_sample_ind = [self.compute_math_accuracy(pred[:,ind], gt[:,ind]) for ind in range (len(predicted_answers[0]))]
        logger.debug(f"Accuracy per sample index: {accuracy_per_sample_ind}")
        
        pass_1 = np.mean(accuracy_per_sample_ind) * 100
        logger.debug(f"Pass@1 for {num_sample_responses}: {pass_1}%")
    
        self.plot_benchmarks(accuracy_per_sample_ind, average_tokens_across_samples)
    
if __name__ == "__main__":
    # Accept args from the user 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_waits", type=int, default=0, help="Number of extra waits to add")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens for the query")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for the query")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for the query")
    parser.add_argument(
        "--dataset_type", 
        type=DatasetType, 
        default=DatasetType.AIME.value, 
        help="Dataset type has to be either aime or math",
        choices=[DatasetType.AIME.value, DatasetType.MATH.value]
    )
    parser.add_argument(
        "--num_sample_responses", 
        type=int, 
        default=4, 
        help="Number of sample responses to generate"
    )
    args = parser.parse_args()
    if args.dataset_type == DatasetType.AIME.value:
        dataset_type = DatasetType.AIME
    elif args.dataset_type == DatasetType.MATH.value:
        dataset_type = DatasetType.MATH

    args = parser.parse_args()
    vllm_client = VLLMClient()

    vllm_client.main(
        num_waits=args.num_waits,
        dataset_type=dataset_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_sample_responses=args.num_sample_responses
    )