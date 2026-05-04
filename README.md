# INT8 Per-Tensor KV Cache Scale Calibration

Generate calibration scales for `--kv-cache-dtype int8_per_tensor` in vLLM.

Loads a model on CPU, processes calibration data layer-by-layer (only 1
layer on GPU at a time), measures K/V activation ranges, and injects
`k_scale`/`v_scale` into the target checkpoint.

**K scales are measured post-RoPE** (after rotary position embeddings),
which matches what actually enters the KV cache. V scales are measured
at v_proj output (RoPE is not applied to V).

## Important: Base vs Quantized

| Quantization | Calibrate from | Why |
|-------------|---------------|-----|
| **GPTQ** | Base model | GPTQ preserves activation ranges |
| **AWQ** (with smoothing) | **Quantized model** (`--quantized`) | AWQ channel-wise scaling changes K/V ranges |

AWQ applies smoothing that rebalances weights between layernorm and
K/V projections. This changes the K activation ranges by up to 15x
compared to the base model. Always use `--quantized` for AWQ models.

## Requirements

- `transformers>=5.7`
- `datasets`
- `safetensors`
- `torch`
- System RAM: ~model_size (17GB for GPTQ W4A16, 54GB for bf16 base)
- GPU VRAM: ~1-3GB (one layer at a time)

## Quick start

```bash
# GPTQ model (calibrate from base):
python calibrate_int8_kv_scales.py --config models/qwen3.6-27b.json

# AWQ model (calibrate from quantized):
python calibrate_int8_kv_scales.py --config models/qwen3-omni-30b-quantized.json
```

## Calibrated models

| Model | Config | Method | K absmax range |
|-------|--------|--------|---------------|
| Qwen3.6-27B GPTQ | `qwen3.6-27b.json` | From base | 9 - 44 |
| Qwen3.6-35B-A3B GPTQ | `qwen3.6-35b-moe.json` | From base | 9 - 18 |
| Qwen3-Omni-30B AWQ | `qwen3-omni-30b-quantized.json` | From quantized | 13 - 156 |

## Examples

### Qwen3.6-27B (GPTQ, from base)

```bash
python calibrate_int8_kv_scales.py --config models/qwen3.6-27b.json
```

### Qwen3.6-35B MoE (GPTQ, from base)

```bash
python calibrate_int8_kv_scales.py --config models/qwen3.6-35b-moe.json
```

### Qwen3-Omni-30B (AWQ, from quantized)

```bash
python calibrate_int8_kv_scales.py --config models/qwen3-omni-30b-quantized.json
```

### Llama 3.1 (from base)

```bash
python calibrate_int8_kv_scales.py --config models/llama-3.1-8b.json
```

### Any model

```bash
# From base model:
python calibrate_int8_kv_scales.py \
  --base-model <HF_model_ID> \
  --target-dir <quantized_checkpoint_dir>

# From quantized model (AWQ):
python calibrate_int8_kv_scales.py --quantized \
  --base-model <local_quantized_dir> \
  --target-dir <local_quantized_dir> \
  --model-class <class_name> \
  --language-model-path <dot.path>
```

## How it works

1. Loads the model on CPU (`device_map=None`). For quantized models,
   patches `compressed_tensors` to skip decompression.
2. Patches `apply_rotary_pos_emb` to capture **post-RoPE K** values
   (dynamically discovers the correct module for any architecture).
3. Registers forward hooks on `v_proj` for V absmax.
4. Embeds all calibration samples (embedding layer on GPU temporarily).
5. For each decoder layer:
   - Moves the layer to GPU
   - Feeds all cached hidden states through it
   - RoPE patch captures K absmax, V hook captures V absmax
   - Moves the layer back to CPU
6. Computes `scale = absmax / 127.0` per layer.
7. Auto-detects the key prefix from the target checkpoint
   (e.g. `model.layers.` vs `model.language_model.layers.`).
8. Injects `{prefix}layers.{i}.self_attn.k_scale` and `v_scale`.

## Config file format

```json
{
    "base_model": "Qwen/Qwen3.6-27B",
    "model_class": "Qwen3_5ForConditionalGeneration",
    "language_model_path": "model.language_model",
    "target_dir": "/models/Qwen3.6-27B-GPTQ-W4A16-G32",
    "num_samples": 512,
    "max_seq_len": 4096
}
```

For quantized models (AWQ or GPTQ):
```json
{
    "base_model": "/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-W4A16",
    "model_class": "Qwen3OmniMoeForConditionalGeneration",
    "language_model_path": "thinker.model",
    "target_dir": "/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-W4A16",
    "num_samples": 512,
    "max_seq_len": 4096,
    "quantized": true
}
```

## Finding `language_model_path` and `model_class`

`language_model_path` is the dot-separated path from the top-level model
to the submodule that contains `embed_tokens`, `layers`, and `rotary_emb`.
`model_class` is the transformers class name for loading.

**Standard CausalLM** (Llama, Qwen2, Mistral, etc.):
- `model_class`: not needed (AutoModelForCausalLM works)
- `language_model_path`: not needed (default: `model`)
- Structure: `model.layers`, `model.embed_tokens`, `model.rotary_emb`

**Qwen3.5 / Qwen3.6 conditional generation** (dense + Mamba):
- `model_class`: `Qwen3_5ForConditionalGeneration`
- `language_model_path`: `model.language_model`
- Structure: `model.language_model.layers`, `model.language_model.embed_tokens`

**Qwen3.5 MoE conditional generation**:
- `model_class`: `Qwen3_5MoeForConditionalGeneration`
- `language_model_path`: `model.language_model`
- Structure: same as above

**Qwen3 Omni MoE** (multimodal: thinker + talker + audio):
- `model_class`: `Qwen3OmniMoeForConditionalGeneration`
- `language_model_path`: `thinker.model`
- Structure: `thinker.model.layers`, `thinker.model.embed_tokens`
- Note: audio/vision towers are automatically excluded from calibration

To find these for a new model:
```bash
# 1. Find model_class from config.json:
python -c "import json; print(json.load(open('config.json'))['architectures'])"
# → ['Qwen3OmniMoeForConditionalGeneration']

# 2. Find language_model_path:
python -c "
import torch
# Use the class from step 1 (or AutoModelForCausalLM for standard models)
from transformers import Qwen3OmniMoeForConditionalGeneration as Cls
model = Cls.from_pretrained('model_id', torch_dtype='auto', device_map=None)
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and len(mod) > 10:
        if hasattr(mod[0], 'self_attn'):
            print(f'language_model_path: {name.rsplit(\".layers\", 1)[0]}')
            break
"
# → language_model_path: thinker.model
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | | JSON config file (see above) |
| `--base-model` | required | HF model ID or local path |
| `--model-class` | `AutoModelForCausalLM` | Transformers class name |
| `--language-model-path` | `None` | Dot path to language model submodule |
| `--target-dir` | required | Checkpoint dir to inject scales into |
| `--quantized` | `False` | Load quantized model (patches compressed_tensors) |
| `--num-samples` | 512 | Number of calibration samples |
| `--max-seq-len` | 4096 | Max sequence length for calibration |
| `--device` | `cuda:0` | GPU device for layer-by-layer processing |

CLI args override config file values.

## After calibration

```bash
vllm serve /models/YourModel --kv-cache-dtype int8_per_tensor ...
```

vLLM automatically loads `k_scale`/`v_scale` from the checkpoint.

## Why post-RoPE matters

Rotary Position Embeddings (RoPE) apply cos/sin rotations to K values.
This can amplify K magnitudes significantly:

| Observation point | Typical K absmax |
|-------------------|-----------------|
| k_proj output (pre-RoPE) | 0.1 - 10 |
| After RoPE (post-RoPE) | 13 - 156 |

Calibrating pre-RoPE gives scales 5-50x too small → massive clipping
at int8 quantization time → garbage output.

## Calibration datasets

Uses a mix of 3 public datasets for representative activation coverage:
- **ultrachat_200k** — general chat
- **Magicoder-Evol-Instruct-110K** — code generation
- **hermes-function-calling-v1** — tool use / function calling

Models without a chat template (e.g. Omni) automatically fall back to
plain text formatting.
