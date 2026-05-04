# INT8 Per-Tensor KV Cache Scale Calibration

Generate calibration scales for `--kv-cache-dtype int8_per_tensor` in vLLM.

Runs the model layer-by-layer (1 layer on GPU at a time), measures
**post-RoPE K** and V activation ranges, and injects `k_scale`/`v_scale`
+ `kv_cache_scheme` into the checkpoint.

## Quick start

```bash
# Recommended: calibrate from the quantized model itself
python calibrate_int8_kv_scales.py --quantized \
  --base-model /models/Qwen3.6-27B-GPTQ-W4A16-G32 \
  --model-class Qwen3_5ForConditionalGeneration \
  --language-model-path model.language_model \
  --target-dir /models/Qwen3.6-27B-GPTQ-W4A16-G32

# Or use a config file:
python calibrate_int8_kv_scales.py --config models/qwen3.6-27b.json
```

## Requirements

- `transformers>=5.7`
- `datasets`, `safetensors`, `torch`
- System RAM: ~model size (17GB for W4A16, 54GB for bf16 base)
- GPU VRAM: ~1-3GB (one layer at a time)

## Config files

```json
{
    "base_model": "/models/Qwen3.6-27B-GPTQ-W4A16-G32",
    "model_class": "Qwen3_5ForConditionalGeneration",
    "language_model_path": "model.language_model",
    "target_dir": "/models/Qwen3.6-27B-GPTQ-W4A16-G32",
    "num_samples": 512,
    "max_seq_len": 4096,
    "quantized": true
}
```

Pre-built configs in `models/`:

| Config | Model | Type |
|--------|-------|------|
| `qwen3.6-27b.json` | Qwen3.6-27B GPTQ | Dense + Mamba |
| `qwen3.6-35b-moe.json` | Qwen3.6-35B-A3B GPTQ | MoE + Mamba |
| `qwen3-omni-30b-quantized.json` | Qwen3-Omni-30B AWQ | Multimodal MoE |
| `llama-3.1-8b.json` | Llama 3.1 8B | Standard CausalLM |

## How it works

1. Loads the model on CPU. For quantized models (`--quantized`),
   patches `compressed_tensors` to handle loading without full decompression.
2. Patches `apply_rotary_pos_emb` to capture **post-RoPE K** values.
   RoPE can amplify K by 5-50x — calibrating pre-RoPE gives wrong scales.
3. Hooks `v_proj` for V absmax (RoPE is not applied to V).
4. Embeds all calibration samples, then for each decoder layer:
   - Moves layer to GPU, runs all samples through it
   - RoPE patch captures K absmax, V hook captures V absmax
   - Moves layer back to CPU
5. Computes `scale = absmax / 127.0` per layer.
6. Injects scales into the checkpoint safetensors.
7. Adds `kv_cache_scheme` to config.json so vLLM loads scales
   via the standard weight loader (same mechanism as FP8).

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | | JSON config file |
| `--base-model` | required | Local path to model (quantized or base) |
| `--quantized` | `False` | Patch compressed_tensors for quantized model loading |
| `--model-class` | `AutoModelForCausalLM` | Transformers class name |
| `--language-model-path` | `None` | Dot path to language model submodule |
| `--target-dir` | required | Checkpoint dir to inject scales into |
| `--num-samples` | 512 | Calibration samples |
| `--max-seq-len` | 4096 | Max sequence length |
| `--device` | `cuda:0` | GPU for layer-by-layer processing |

## Finding `language_model_path` and `model_class`

**Standard CausalLM** (Llama, Qwen2, Mistral):
- `model_class`: not needed
- `language_model_path`: not needed

**Qwen3.5 / Qwen3.6** (dense + Mamba):
- `model_class`: `Qwen3_5ForConditionalGeneration`
- `language_model_path`: `model.language_model`

**Qwen3.5 MoE**:
- `model_class`: `Qwen3_5MoeForConditionalGeneration`
- `language_model_path`: `model.language_model`

**Qwen3 Omni MoE** (multimodal):
- `model_class`: `Qwen3OmniMoeForConditionalGeneration`
- `language_model_path`: `thinker.model`

For other models:
```bash
# 1. model_class:
python -c "import json; print(json.load(open('config.json'))['architectures'])"

# 2. language_model_path:
python -c "
import torch
from transformers import <ModelClass>  # from step 1
model = <ModelClass>.from_pretrained('model_path', dtype='auto', device_map=None)
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and len(mod) > 10:
        if hasattr(mod[0], 'self_attn'):
            print(name.rsplit('.layers', 1)[0])
            break
"
```

## After calibration

```bash
vllm serve /models/YourModel --kv-cache-dtype int8_per_tensor ...
```

vLLM loads `k_scale`/`v_scale` automatically via `kv_cache_scheme` in config.json.
