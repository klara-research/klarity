from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LogitsProcessorList
from PIL import Image
from klarity import UncertaintyEstimator
from klarity.core.analyzer import VLMAnalyzer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize VLM model
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, output_attentions=True, low_cpu_mem_usage=True)

processor = AutoProcessor.from_pretrained(model_id)

# Create estimator with VLMAnalyzer
estimator = UncertaintyEstimator(
    top_k=100,
    analyzer=VLMAnalyzer(
        min_token_prob=0.01,
        insight_model="together:meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        insight_api_key="your_api_key",
        vision_config=model.config.vision_config,
        use_cls_token=True,
    ),
)
uncertainty_processor = estimator.get_logits_processor()

# Set up generation
image_path = "examples/images/plane.jpg"
question = "How many engines does the plane have?"
image = Image.open(image_path)

# Prepare input with image and text
conversation = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]}]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process inputs
inputs = processor(images=image, text=prompt, return_tensors="pt")

try:
    # Generate with uncertainty and attention analysis
    generation_output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        logits_processor=LogitsProcessorList([uncertainty_processor]),
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
        use_cache=True,
    )

    # Analyze the generation - Note the corrected parameter order
    result = estimator.analyze_generation(
        generation_output=generation_output,
        model=model,  # Add model parameter
        tokenizer=processor,  # Changed from processor to tokenizer parameter
        processor=uncertainty_processor,
        prompt=question,
    )

    # Get generated text using the processor instead of raw sequences
    input_length = inputs.input_ids.shape[1]
    generated_sequence = generation_output.sequences[0][input_length:]
    generated_text = processor.decode(generated_sequence, skip_special_tokens=True)

    print(f"\nQuestion: {question}")
    print(f"Generated answer: {generated_text}")

    # Token Analysis
    print("\nDetailed Token Analysis:")
    for idx, metrics in enumerate(result.token_metrics):
        print(f"\nStep {idx}:")
        print(f"Raw entropy: {metrics.raw_entropy:.4f}")
        print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
        print("Top 3 predictions:")
        for i, pred in enumerate(metrics.token_predictions[:3], 1):
            print(f"  {i}. {pred.token} (prob: {pred.probability:.4f})")

    # Attention Analysis (if available)
    if result.attention_data:
        print("\nAttention Analysis:")
        estimator.analyzer.visualize_attention(
            result.attention_data, image, save_path="examples/attention_maps/attention_visualization.png"
        )

    # Show comprehensive insight
    print("\nComprehensive Analysis:")
    print(result.overall_insight)

except Exception as e:
    print(f"Error during generation: {str(e)}")
    import traceback

    traceback.print_exc()  # Print full traceback for debugging
