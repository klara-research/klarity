import unittest
import numpy as np
from unittest.mock import Mock, patch
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.klarity.benchmarks.reasoning_budget_control import VLLMClient


class LogProb:
    def __init__(self, logprob: float):
        self.logprob = logprob


class MockOutput:
    def __init__(self, text: str, token_ids: List[int], num_generated_tokens: int, logprobs=None):
        self.text = text
        self.token_ids = token_ids
        self.num_generated_tokens = num_generated_tokens
        if logprobs is None:
            # Create default logprobs
            self.logprobs = [{i: LogProb(-1.0) for i in range(5)} for _ in token_ids]
        else:
            self.logprobs = logprobs


class MockOutputGeneration:
    def __init__(self, text: str, token_ids: List[int], num_generated_tokens: int, logprobs=None):
        self.outputs = [MockOutput(text, token_ids, num_generated_tokens, logprobs)]


class TestVLLMClient(unittest.TestCase):
    def setUp(self):
        # Mock the LLM and tokenizer
        self.mock_llm = Mock(spec=LLM)
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.apply_chat_template = Mock(return_value=[1, 2, 3, 4])
        self.mock_tokenizer.decode = Mock(return_value="<s>test</s>")

        # Create patches
        self.llm_patcher = patch("src.klarity.benchmarks.reasoning_budget_control.LLM", return_value=self.mock_llm)
        self.tokenizer_patcher = patch("src.klarity.benchmarks.reasoning_budget_control.AutoTokenizer")

        # Start patches
        self.llm_mock = self.llm_patcher.start()
        self.tokenizer_mock = self.tokenizer_patcher.start()
        self.tokenizer_mock.from_pretrained.return_value = self.mock_tokenizer

        # Initialize client with mocked dependencies
        self.client = VLLMClient()

    def tearDown(self):
        # Stop patches
        self.llm_patcher.stop()
        self.tokenizer_patcher.stop()

    def test_extract_answer_valid(self):
        """Test extract_answer with valid input"""
        test_cases = [
            ("The answer is boxed{42}", "42"),
            ("Multiple boxed{1} answers boxed{2}", "2"),
            ("No boxes", ""),
            ("Incomplete boxed{", ""),
            ("boxed{nested boxed{42}}", "42"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.client.extract_answer(input_text)
                self.assertEqual(result, expected)

    def test_compute_math_accuracy(self):
        """Test compute_math_accuracy with various test cases"""
        test_cases = [
            (["42", "7"], ["42", "7"], 1.0),  # All correct
            (["42", "8"], ["42", "7"], 0.5),  # Half correct
            ([], [], 0.0),  # Empty lists
        ]

        for predictions, ground_truths, expected_accuracy in test_cases:
            with self.subTest(predictions=predictions, ground_truths=ground_truths):
                accuracy = self.client.compute_math_accuracy(predictions, ground_truths)
                self.assertEqual(accuracy, expected_accuracy)

    def test_compute_math_accuracy_different_lengths(self):
        """Test that compute_math_accuracy raises ValueError when lists have different lengths"""
        predictions = ["42", "7"]
        ground_truths = ["42", "7", "8"]

        with self.assertRaises(ValueError) as _:
            self.client.compute_math_accuracy(predictions, ground_truths)

    def test_add_system_prompt(self):
        """Test add_system_prompt functionality"""
        test_prompt = "What is 2+2?"
        result = self.client.add_system_prompt(test_prompt)
        desired_result = (
            "Please reason step by step, and put your final answer within \\boxed\{\}."
            "Solve the following problem:" + "What is 2+2?"
        )
        self.assertEqual(result, desired_result)

    def test_tokenize_prompt_basic(self):
        """Test basic tokenize_prompt functionality"""
        test_prompt = "What is 2+2?"
        # Set up mock return values
        expected_tokens = [1, 2, 3, 4]
        expected_decoded = "<s>What is 2+2?</s>"
        self.mock_tokenizer.apply_chat_template.return_value = expected_tokens
        self.mock_tokenizer.decode.return_value = expected_decoded

        result = self.client.tokenize_prompt(test_prompt)

        # Verify the tokenizer was called correctly
        system_prompt = self.client.add_system_prompt(test_prompt)
        self.mock_tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": system_prompt}],
            add_generation_prompt=True
        )
        self.mock_tokenizer.decode.assert_called_once_with(expected_tokens, skip_special_tokens=False)

        # Verify the final result
        expected_result = expected_decoded + self.client.THINK_START_STR
        self.assertEqual(result, expected_result)

    def test_tokenize_prompt_empty(self):
        """Test tokenize_prompt with empty prompt"""
        test_prompt = ""
        # Set up mock return values
        expected_tokens = [1, 2]
        expected_decoded = "<s></s>"
        self.mock_tokenizer.apply_chat_template.return_value = expected_tokens
        self.mock_tokenizer.decode.return_value = expected_decoded

        result = self.client.tokenize_prompt(test_prompt)

        # Verify the final result
        expected_result = expected_decoded + self.client.THINK_START_STR
        self.assertEqual(result, expected_result)

    def test_tokenize_prompt_special_characters(self):
        """Test tokenize_prompt with special characters"""
        test_prompt = "What is 2²?"
        # Set up mock return values
        expected_tokens = [1, 2, 3, 4, 5]
        expected_decoded = "<s>What is 2²?</s>"
        self.mock_tokenizer.apply_chat_template.return_value = expected_tokens
        self.mock_tokenizer.decode.return_value = expected_decoded

        result = self.client.tokenize_prompt(test_prompt)

        # Verify the final result
        expected_result = expected_decoded + self.client.THINK_START_STR
        self.assertEqual(result, expected_result)

    def test_compute_metrics_basic(self):
        """Test compute_metrics with perfect predictions"""
        # Test data
        total_generated_tokens = [[10, 15], [12, 18]]  # 2 samples, 2 responses each
        predicted_answers = [["4", "4"], ["16", "16"]]  # 2 samples, 2 responses each
        gt_answers = ["4", "16"]  # 2 samples, 2 responses each

        with patch.object(self.client, "plot_benchmarks"):
            with patch.object(self.client, "compute_math_accuracy", return_value=1.0):
                # Call the function
                accuracy_per_sample, avg_tokens, pass_1 = self.client.compute_metrics(
                    total_generated_tokens=total_generated_tokens,
                    predicted_answers=predicted_answers,
                    gt_answers=gt_answers,
                )

                # Verify accuracies
                self.assertEqual(len(accuracy_per_sample), 2)  # Two samples
                self.assertTrue(all(acc == 1.0 for acc in accuracy_per_sample))

                # Verify average tokens
                expected_avg_tokens = np.array([11, 16.5])  # Mean of [10,12] and [15,18]
                np.testing.assert_array_almost_equal(avg_tokens, expected_avg_tokens)

                # Verify pass@1
                self.assertEqual(pass_1, 100.0)

    def test_compute_metrics_with_errors(self):
        """Test compute_metrics with some incorrect predictions"""
        # Test data with some incorrect predictions
        total_generated_tokens = [[10, 15], [12, 18]]
        predicted_answers = [["4", "5"], ["16", "15"]]  # Some wrong answers
        gt_answers = ["4", "16"]

        with patch.object(self.client, "plot_benchmarks"):
            with patch.object(self.client, "compute_math_accuracy", return_value=0.5):
                accuracy_per_sample, avg_tokens, pass_1 = self.client.compute_metrics(
                    total_generated_tokens=total_generated_tokens,
                    predicted_answers=predicted_answers,
                    gt_answers=gt_answers,
                )

                # Verify accuracies
                self.assertEqual(len(accuracy_per_sample), 2)  # Two samples
                self.assertTrue(all(acc == 0.5 for acc in accuracy_per_sample))

                # Verify average tokens
                expected_avg_tokens = np.array([11, 16.5])  # Mean of [10,12] and [15,18]
                np.testing.assert_array_almost_equal(avg_tokens, expected_avg_tokens)

                # Since accuracy is 0.5 for both samples, pass@1 should be 50%
                self.assertEqual(pass_1, 50.0)

    def test_compute_metrics_input_validation(self):
        """Test compute_metrics with invalid inputs"""
        with patch.object(self.client, "plot_benchmarks"):
            # Test empty inputs
            with self.assertRaises(ValueError):
                self.client.compute_metrics([], [], [])

    def test_compute_metrics_mixed_predictions(self):
        """Test compute_metrics with a mix of correct and incorrect predictions"""
        total_generated_tokens = [[10, 15], [12, 18], [20, 25]]  # 3 prompts, 2 responses each
        predicted_answers = [["4", "5"], ["16", "16"], ["25", "24"]]  # Some wrong answers
        gt_answers = ["4", "16", "25"]

        with patch.object(self.client, "plot_benchmarks"):
            accuracies, avg_tokens, pass_1 = self.client.compute_metrics(
                total_generated_tokens=total_generated_tokens,
                predicted_answers=predicted_answers,
                gt_answers=gt_answers,
            )

            # Check accuracies
            expected_accuracies = [1.0, 1 / 3]  # First and last samples have one wrong answer
            np.testing.assert_array_almost_equal(accuracies, expected_accuracies)

            # Check average tokens
            expected_avg_tokens = np.array([14, 19.333333])  # Average of each pair
            np.testing.assert_array_almost_equal(avg_tokens, expected_avg_tokens)

            # Check pass@1 score (average of accuracies * 100)
            self.assertEqual(pass_1, (1.0 + (1 / 3)) / 2 * 100)

    def test_get_vllm_output_single_prompt_single_sample(self):
        """Test get_vllm_output with a single prompt and single sample"""
        mock_prompt_output = MockOutputGeneration("Test text", [1, 2, 3], 3)

        generated_texts, num_tokens, logprobs, token_ids = self.client.get_vllm_output([mock_prompt_output])

        self.assertEqual(generated_texts, ["Test text"])
        self.assertEqual(num_tokens, [3])
        self.assertEqual(len(logprobs[0]), 3)  # One logprob per token
        self.assertEqual(token_ids, [[1, 2, 3]])

    def test_get_vllm_output_multiple_prompts(self):
        """Test get_vllm_output with multiple prompts"""
        mock_outputs = [MockOutputGeneration("Prompt 1", [1, 2], 2), MockOutputGeneration("Prompt 2", [3, 4, 5], 3)]

        generated_texts, num_tokens, logprobs, token_ids = self.client.get_vllm_output(mock_outputs)

        self.assertEqual(generated_texts, ["Prompt 1", "Prompt 2"])
        self.assertEqual(num_tokens, [2, 3])
        self.assertEqual(len(logprobs), 2)  # Two sets of logprobs
        self.assertEqual(token_ids, [[1, 2], [3, 4, 5]])

    def test_get_vllm_output_empty_outputs(self):
        """Test get_vllm_output with empty outputs"""
        generated_texts, num_tokens, logprobs, token_ids = self.client.get_vllm_output([])

        self.assertEqual(generated_texts, [])
        self.assertEqual(num_tokens, [])
        self.assertEqual(logprobs, [])
        self.assertEqual(token_ids, [])

    def test_get_vllm_output_multiple_prompts_multiple_samples(self):
        """Test get_vllm_output with multiple prompts, each having multiple samples"""
        # Create mock outputs for first prompt with 2 samples
        mock_output1_1 = MockOutput("P1S1", [1, 2], 2)
        mock_output1_2 = MockOutput("P1S2", [3, 4], 2)
        mock_prompt1 = MockOutputGeneration("P1S1", [1, 2], 2)
        mock_prompt1.outputs = [mock_output1_1, mock_output1_2]

        # Create mock outputs for second prompt with 2 samples
        mock_output2_1 = MockOutput("P2S1", [5, 6], 2)
        mock_output2_2 = MockOutput("P2S2", [7, 8, 9], 3)
        mock_prompt2 = MockOutputGeneration("P2S1", [5, 6], 2)
        mock_prompt2.outputs = [mock_output2_1, mock_output2_2]

        # Test with both prompts
        generated_texts, num_tokens, logprobs, token_ids = self.client.get_vllm_output([mock_prompt1, mock_prompt2])

        # Verify the results
        self.assertEqual(generated_texts, ["P1S1", "P1S2", "P2S1", "P2S2"])
        self.assertEqual(num_tokens, [2, 2, 2, 3])
        self.assertEqual(len(logprobs), 4)  # Four sets of logprobs
        self.assertEqual(token_ids, [[1, 2], [3, 4], [5, 6], [7, 8, 9]])
        self.assertEqual(num_tokens, [2, 2, 2, 3])

    def test_find_first_subsequence(self):
        """Test finding first occurrence of a subsequence"""
        test_cases = [
            ([1, 2, 3, 4, 2, 3], [2, 3], 1),  # Basic case
            ([1, 2, 3], [4], -1),  # Subsequence not found
            ([1, 2, 2], [2], 1),  # Multiple occurrences, should find first
            ([], [1], -1),  # Empty array
            ([1, 2, 3], [], -1),  # Empty subsequence
            ([1, 2, 3, 4, 5], [3, 4, 5], 2),  # Subsequence at end
            ([1, 2, 3], [1, 2, 3], 0),  # Full sequence match
        ]

        for arr, sub, expected in test_cases:
            with self.subTest(arr=arr, sub=sub):
                result = self.client.find_first_or_last_subsequence(arr, sub, is_first=True)
                self.assertEqual(result, expected)

    def test_find_last_subsequence(self):
        """Test finding last occurrence of a subsequence"""
        test_cases = [
            ([1, 2, 3, 4, 2, 3], [2, 3], 4),  # Basic case
            ([1, 2, 3], [4], -1),  # Subsequence not found
            ([1, 2, 2], [2], 2),  # Multiple occurrences, should find last
            ([], [1], -1),  # Empty array
            ([1, 2, 3], [], -1),  # Empty subsequence
            ([1, 2, 3, 4, 5], [1, 2, 3], 0),  # Subsequence at start
            ([1, 2, 3], [1, 2, 3], 0),  # Full sequence match
        ]

        for arr, sub, expected in test_cases:
            with self.subTest(arr=arr, sub=sub):
                result = self.client.find_first_or_last_subsequence(arr, sub, is_first=False)
                self.assertEqual(result, expected)

    def test_compute_entropy_of_answer(self):
        """Test computing entropy of answer"""

        # Mock the tokenizer's behavior for boxed strings
        def mock_tokenize(text):
            if text == self.client.BOXED_START_STR:
                return {"input_ids": [0, 10, 11]}  # Start token + boxed tokens
            elif text == self.client.BOXED_END_STR:
                return {"input_ids": [0, 12]}  # Start token + end token
            return {"input_ids": [0, 1, 2]}  # Default case

        self.mock_tokenizer.side_effect = mock_tokenize

        # Test case 1: No boxed content
        response_token_ids = tuple([1, 2, 3, 4, 5])
        response_logprobs = np.zeros((len(response_token_ids), 10))
        result = self.client.compute_entropy_of_answer(response_token_ids, response_logprobs)
        self.assertEqual(result, 0.0)

        # Test case 2: With boxed content
        response_token_ids = tuple([1, 2, 10, 11, 3, 4, 12, 5])  # Contains boxed{3, 4}
        response_logprobs = np.ones((len(response_token_ids), 10)) * np.log(0.1)  # Equal probabilities
        result = self.client.compute_entropy_of_answer(response_token_ids, response_logprobs)
        self.assertGreater(result, 0.0)  # Entropy should be positive


if __name__ == "__main__":
    unittest.main()
