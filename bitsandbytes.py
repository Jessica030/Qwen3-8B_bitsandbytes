import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import gc
from huggingface_hub import HfApi

model = "Qwen/Qwen3-8B"  # Example model, can be changed as needed
# Shared tokenizer
def initialize_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer initialized successfully!")
        return tokenizer
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return None

def load_original_model(model_name, device):
    """Load full-precision model (fp16)."""
    print("Loading original (fp16) model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        print("Original model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading original model: {e}")
        return None

def load_quantized_model(model_name, device):
    """Load 8-bit quantized model."""
    print("Loading quantized (8-bit) model...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold = 0.0,
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()
        print("Quantized model loaded successfully!")
        return model, bnb_config
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        return None, None

def measure_latency_and_memory(model, tokenizer, prompt, device, max_new_tokens=128, num_runs=10):
    """
    Measure average latency (s) and peak memory (MB) over multiple runs.
    
    Args:
        model: Loaded model.
        tokenizer: Shared tokenizer.
        prompt: Input prompt string.
        device: Device to run on (e.g., 'cuda:1').
        max_new_tokens: Tokens to generate.
        num_runs: Number of iterations for averaging.
    
    Returns:
        Dict with 'avg_latency', 'std_latency', 'tps', 'peak_memory_mb'.
    """
    if model is None or tokenizer is None:
        print("Error: Model or tokenizer is None.")
        return None

    # Validate device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        return None
    if device.startswith("cuda"):
        device_idx = int(device.split(":")[1]) if ":" in device else 0
        if device_idx >= torch.cuda.device_count():
            print(f"Error: Device {device} is invalid. Available GPUs: {torch.cuda.device_count()}")
            return None

    model.eval()
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"Inputs allocated on {device}")
    except Exception as e:
        print(f"Error tokenizing prompt: {e}")
        return None
    
    latencies = []
    peak_memories = []
    
    # Warmup run
    try:
        print("Starting warmup run...")
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()
        print("Warmup run completed.")
    except Exception as e:
        print(f"Error during warmup run: {e}")
        return None
    
    for run in range(num_runs):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        try:
            print(f"Starting run {run + 1}/{num_runs}...")
            start_event.record()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            end_event.record()
            torch.cuda.synchronize()
            
            latency = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
            latencies.append(latency)
            
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
            peak_memories.append(peak_memory)
            
            print(f"Run {run + 1}/{num_runs}: Latency={latency:.3f}s, Peak Mem={peak_memory:.2f}MB, Current Mem={current_memory:.2f}MB")
        except Exception as e:
            print(f"Error during run {run + 1}: {e}")
            return None
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    tokens_generated = max_new_tokens
    tps = tokens_generated / avg_latency if avg_latency > 0 else 0
    avg_peak_memory = np.mean(peak_memories)
    
    print(f"\nAverages: Latency={avg_latency:.3f}s (±{std_latency:.3f}s), TPS={tps:.2f}, Peak Mem={avg_peak_memory:.2f}MB")
    
    return {
        "avg_latency": avg_latency,
        "std_latency": std_latency,
        "tps": tps,
        "peak_memory_mb": avg_peak_memory,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM quantization performance")
    parser.add_argument("--model", default=model, help="Model name from Hugging Face")
    parser.add_argument("--prompt", default="Explain the benefits of renewable energy in simple terms.", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs for averaging")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu", help="Device to run on")
    args = parser.parse_args()
    
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(args.model)
    if tokenizer is None:
        return
    
    # Evaluate original model
    print("\n=== Evaluating Original (fp16) Model ===")
    original_model = load_original_model(args.model, args.device)
    if original_model is None:
        return
    original_results = measure_latency_and_memory(original_model, tokenizer, args.prompt, args.device, args.max_new_tokens, args.num_runs)
    
    # Cleanup
    del original_model
    gc.collect()
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    
    # Evaluate quantized model
    print("\n=== Evaluating and Saving Quantized (8-bit) Model ===")
    quantized_model, _ = load_quantized_model(args.model, args.device)  # ← unpack here
    if quantized_model is None:
        return

    # Save the quantized model
    save_dir = "./qwen3-8B-8bit"
    print(f"Saving quantized model to {save_dir}...")
    quantized_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Quantized model and tokenizer saved!")

    # 🔥 PUSH TO HUB
    repo_id = "Jessica030/Qwen3-8B-8bit"  # e.g., "jessicang/Qwen3-8B-4bit"
    print(f"Pushing to Hugging Face Hub: {repo_id}")

    quantized_model.push_to_hub(repo_id, use_auth_token=True)
    tokenizer.push_to_hub(repo_id, use_auth_token=True)

    print("✅ Model and tokenizer successfully pushed to Hugging Face!")

    quantized_results = measure_latency_and_memory(quantized_model, tokenizer, args.prompt, args.device, args.max_new_tokens, args.num_runs)
    
    # Cleanup
    del quantized_model
    gc.collect()
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    
    # Comparisons
    if original_results and quantized_results:
        latency_delta = quantized_results["avg_latency"] - original_results["avg_latency"]
        #latency_speedup_pct = (original_results["avg_latency"] / quantized_results["avg_latency"] - 1) * 100 if original_results["avg_latency"] > 0 else 0
        latency_speedup_pct = (quantized_results["avg_latency"] - original_results["avg_latency"])/(original_results["avg_latency"]) * 100 if original_results["avg_latency"] > 0 else 0
        tps_improvement_pct = (quantized_results["tps"] / original_results["tps"] - 1) * 100 if original_results["tps"] > 0 else 0
        tps_improvement = original_results["tps"] - quantized_results["tps"]
        memory_delta_mb = quantized_results["peak_memory_mb"] - original_results["peak_memory_mb"]
        memory_reduction_pct = (quantized_results["peak_memory_mb"] - original_results["peak_memory_mb"])/(original_results["peak_memory_mb"]) * 100 if original_results["peak_memory_mb"] > 0 else 0
        
        print("\n=== Comparison Summary ===")
        print(f"Original: Latency={original_results['avg_latency']:.3f}s, TPS={original_results['tps']:.2f}, Peak Mem={original_results['peak_memory_mb']:.2f}MB")
        print(f"Quantized: Latency={quantized_results['avg_latency']:.3f}s, TPS={quantized_results['tps']:.2f}, Peak Mem={quantized_results['peak_memory_mb']:.2f}MB")
        print(f"\nLatency Delta: {latency_delta:+.3f}s (Difference: {latency_speedup_pct:+.1f}%)")
        print(f"TPS Improvement: {tps_improvement:.1f} (Difference: {tps_improvement_pct:+.1f}%)")
        print(f"Memory Delta: {memory_delta_mb:+.2f}MB (Difference: {memory_reduction_pct:+.1f}%)")

        if latency_speedup_pct < 0:
            print(f"\nVerdict: Quantization provides {latency_speedup_pct:.1f}% latency speedup and {memory_reduction_pct:.1f}% memory savings!")
        else:
            print("\nVerdict: Quantization may have overhead—try a longer generation or different hardware.")
    else:
        print("\nComparison skipped due to errors in evaluation.")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()