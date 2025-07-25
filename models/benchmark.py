import os
import time
import torch
import numpy as np
from models.llm_refinement import LLMRefiner
from transformers import GPT2Model, GPT2Config
# For LLaMA-2, Falcon, and GPT-3.5, use HuggingFace or API stubs as placeholders
try:
    from ptflops import get_model_complexity_info
    PT_FLOPS = True
except ImportError:
    PT_FLOPS = False

def benchmark_model(model, input_shape, device='cuda', n_runs=10):
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    # Timing
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    end = time.perf_counter()
    end_mem = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    avg_time = (end - start) / n_runs
    mem_usage = (end_mem - start_mem) / (1024 ** 2)  # MB
    # FLOPs
    flops = None
    if PT_FLOPS:
        with torch.cuda.device(device):
            flops, params = get_model_complexity_info(model, input_shape[1:], as_strings=False, print_per_layer_stat=False)
    return avg_time, mem_usage, flops

def print_results_table(results):
    print("| Model         | Time (s) | Mem (MB) | FLOPs      |")
    print("|--------------|----------|----------|------------|")
    for name, (t, m, f) in results.items():
        f_str = f"{f/1e9:.2f}G" if f is not None else "-"
        print(f"| {name:<12} | {t:.4f}   | {m:.2f}     | {f_str:<10} |")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = 256
    seq_len = 8
    batch_size = 4
    input_shape = (batch_size, seq_len, input_dim)
    results = {}
    # Your LLM-based refiner
    model = LLMRefiner(input_dim)
    t, m, f = benchmark_model(model, input_shape, device=device)
    results['ConLLM'] = (t, m, f)
    # GPT-2 baseline
    gpt2 = GPT2Model(GPT2Config(n_embd=input_dim, n_layer=2, n_head=8, n_positions=seq_len, n_ctx=seq_len, vocab_size=1000))
    t, m, f = benchmark_model(gpt2, (batch_size, seq_len, input_dim), device=device)
    results['GPT-2'] = (t, m, f)
    # Placeholders for GPT-3.5, LLaMA-2, Falcon (API or large models not run locally)
    results['GPT-3.5'] = (0.0, 0.0, None)
    results['LLaMA-2'] = (0.0, 0.0, None)
    results['Falcon'] = (0.0, 0.0, None)
    print_results_table(results)
    print("Note: GPT-3.5, LLaMA-2, Falcon results are placeholders (API/large model not run locally)") 