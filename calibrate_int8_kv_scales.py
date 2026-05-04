"""
Calibrate INT8 per-tensor KV cache scales for any model.

Loads a model on CPU, runs forward passes layer-by-layer (one layer on
GPU at a time), measures post-RoPE K absmax and V absmax per layer,
then injects k_scale/v_scale into the target checkpoint.

Supports both base (bf16) and quantized (GPTQ/AWQ) models via the
--quantized flag. AWQ models with smoothing MUST be calibrated from
the quantized model (not the base) because AWQ changes K/V ranges.

K scales are measured AFTER rotary position embeddings (post-RoPE),
which is what actually enters the KV cache. V scales are measured at
v_proj output (V is not rotated by RoPE).

Requirements: ~model_size system RAM, ~1 layer on GPU (~1-3GB).

Usage:
  # From base model (GPTQ preserves activation ranges):
  python calibrate_int8_kv_scales.py \
    --base-model Qwen/Qwen3.6-27B \
    --model-class Qwen3_5ForConditionalGeneration \
    --language-model-path model.language_model \
    --target-dir /models/Qwen3.6-27B-GPTQ-W4A16-G32

  # From quantized model (AWQ changes activation ranges):
  python calibrate_int8_kv_scales.py --quantized \
    --base-model /models/Qwen3-Omni-30B-A3B-Instruct-AWQ-W4A16 \
    --model-class Qwen3OmniMoeForConditionalGeneration \
    --language-model-path thinker.model \
    --target-dir /models/Qwen3-Omni-30B-A3B-Instruct-AWQ-W4A16

See README.md for more examples.
"""

import argparse
import gc
import json
import os
from collections import defaultdict

import torch
from datasets import concatenate_datasets, load_dataset
from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate INT8 per-tensor KV cache scales")
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file (overrides other args)")
    p.add_argument("--base-model", type=str, default=None,
                   help="HuggingFace model ID or local path to base (bf16) model")
    p.add_argument("--model-class", type=str, default=None,
                   help="Transformers class name if AutoModel doesn't work "
                        "(e.g. Qwen3_5ForConditionalGeneration)")
    p.add_argument("--language-model-path", type=str, default=None,
                   help="Dot-separated path to the language model inside the "
                        "top-level model (e.g. 'model.language_model' for "
                        "conditional generation models). If not set, assumes "
                        "standard CausalLM layout (model.layers)")
    p.add_argument("--target-dir", type=str, default=None,
                   help="Checkpoint dir to inject scales into (e.g. GPTQ dir)")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--k-proj-suffix", type=str, default=".self_attn.k_proj",
                   help="Module name suffix for K projection")
    p.add_argument("--v-proj-suffix", type=str, default=".self_attn.v_proj",
                   help="Module name suffix for V projection")
    p.add_argument("--quantized", action="store_true",
                   help="Load a quantized model (GPTQ/AWQ) instead of base bf16. "
                        "Use --base-model to point to the quantized checkpoint. "
                        "Patches compressed_tensors to handle loading.")
    args = p.parse_args()

    # Load config file and merge (config values override defaults,
    # CLI explicit args override config)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        key_map = {
            "base_model": "base_model",
            "model_class": "model_class",
            "language_model_path": "language_model_path",
            "target_dir": "target_dir",
            "num_samples": "num_samples",
            "max_seq_len": "max_seq_len",
            "device": "device",
            "k_proj_suffix": "k_proj_suffix",
            "v_proj_suffix": "v_proj_suffix",
            "quantized": "quantized",
        }
        for json_key, attr in key_map.items():
            if json_key in cfg and getattr(args, attr) is None:
                setattr(args, attr, cfg[json_key])
            elif json_key in cfg and attr in ("num_samples", "max_seq_len"):
                # For int defaults, config overrides only if CLI wasn't set
                if attr == "num_samples" and args.num_samples == 512:
                    args.num_samples = cfg[json_key]
                elif attr == "max_seq_len" and args.max_seq_len == 4096:
                    args.max_seq_len = cfg[json_key]

    if not args.base_model:
        p.error("--base-model is required (or set in --config)")
    if not args.target_dir:
        p.error("--target-dir is required (or set in --config)")

    return args


def load_calibration_data(tokenizer, num_samples, max_seq_len):
    """Load mixed calibration dataset (chat + code + function calling)."""
    samples_per_ds = num_samples // 3

    ds_ultrachat = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{samples_per_ds}]",
    ).select_columns(["messages"])

    ds_code = load_dataset(
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        split=f"train[:{samples_per_ds}]",
    )
    ds_code = ds_code.map(
        lambda ex: {"messages": [
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]},
        ]},
        remove_columns=ds_code.column_names,
    )

    ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}
    ds_hermes = load_dataset(
        "NousResearch/hermes-function-calling-v1",
        split=f"train[:{samples_per_ds}]",
    )
    ds_hermes = ds_hermes.map(
        lambda ex: {"messages": [
            {"role": ROLE_MAP[t["from"]], "content": t["value"]}
            for t in ex["conversations"] if t["from"] in ROLE_MAP
        ]},
        remove_columns=ds_hermes.column_names,
    )
    ds_hermes = ds_hermes.filter(lambda x: len(x["messages"]) > 0)

    ds = concatenate_datasets([ds_ultrachat, ds_code, ds_hermes])
    ds = ds.shuffle(seed=42)

    def tokenize(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False)
        except (ValueError, AttributeError):
            # Fallback for models without chat template (e.g. Omni)
            text = "\n".join(
                f"{m['role']}: {m['content']}" for m in example["messages"])
        return tokenizer(
            text, padding=False, max_length=max_seq_len,
            truncation=True, add_special_tokens=False)

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def _patch_compressed_tensors():
    """Patch compressed_tensors to handle quantized model loading.

    Fixes group_size=0 validation and skips decompression that crashes
    on some GPTQ/AWQ checkpoints. The quantized Linears do dequant
    on-the-fly in forward() so decompression isn't needed.
    """
    try:
        from compressed_tensors.quantization import QuantizationArgs
        _orig_v = QuantizationArgs.__pydantic_validator__

        class _PV:
            def __init__(self, o):
                self._o = o
            def validate_python(self, d, *a, **k):
                if isinstance(d, dict) and d.get('group_size') == 0:
                    d = dict(d)
                    d['group_size'] = -1
                    d['strategy'] = 'channel'
                return self._o.validate_python(d, *a, **k)
            def __getattr__(self, n):
                return getattr(self._o, n)

        QuantizationArgs.__pydantic_validator__ = _PV(_orig_v)

        from compressed_tensors.compressors.model_compressors.model_compressor import (
            ModelCompressor,
        )
        ModelCompressor.decompress_model = lambda self, model: None
    except ImportError:
        pass


def load_model(base_model, model_class, quantized=False):
    """Load model on CPU."""
    if quantized:
        _patch_compressed_tensors()

    if model_class:
        import transformers
        cls = getattr(transformers, model_class)
        model = cls.from_pretrained(
            base_model, torch_dtype="auto", device_map=None)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype="auto", device_map=None,
            trust_remote_code=True)

    if quantized:
        # Clear decompression hooks that would crash on forward
        model._forward_pre_hooks.clear()
        for m in model.modules():
            m._forward_pre_hooks.clear()

    return model


def resolve_language_model(model, lm_path):
    """Navigate to the language model submodule."""
    if lm_path is None:
        return model.model
    obj = model
    for attr in lm_path.split("."):
        obj = getattr(obj, attr)
    return obj


def extract_layer_idx(name):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def register_hooks(model, k_suffix, v_suffix):
    """Register hooks to capture post-RoPE K and raw V absmax.

    Hooks apply_rotary_pos_emb to capture K AFTER rotary embeddings
    (which is what actually enters the KV cache). V is captured at
    v_proj output since RoPE is not applied to V.

    Falls back to k_proj output if apply_rotary_pos_emb is not found.
    """
    k_absmax = defaultdict(float)
    v_absmax = defaultdict(float)
    hooks = []
    _current_layer_idx = [None]

    # Strategy 1: Patch apply_rotary_pos_emb to capture post-RoPE K
    rope_patched = False
    rope_modules = set()

    # Find all modules that define apply_rotary_pos_emb by inspecting
    # the actual attention classes used in the model.
    import inspect
    seen_classes = set()
    for name, module in model.named_modules():
        cls = type(module)
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        attn_module = inspect.getmodule(cls)
        if attn_module and hasattr(attn_module, 'apply_rotary_pos_emb'):
            rope_modules.add(attn_module)

    for mod in rope_modules:
        _orig_fn = mod.apply_rotary_pos_emb

        def _make_patched(orig):
            def _patched(q, k, cos, sin, *args, **kwargs):
                result = orig(q, k, cos, sin, *args, **kwargs)
                q_rot, k_rot = result
                idx = _current_layer_idx[0]
                if idx is not None:
                    with torch.no_grad():
                        k_absmax[idx] = max(
                            k_absmax[idx],
                            k_rot.float().abs().max().item())
                return result
            return _patched

        mod.apply_rotary_pos_emb = _make_patched(_orig_fn)
        rope_patched = True

    if rope_patched:
        print("  Using post-RoPE K observation (patched apply_rotary_pos_emb)")
    else:
        print("  WARNING: apply_rotary_pos_emb not found, using pre-RoPE K "
              "(scales may be inaccurate)")

    # Hook v_proj for V absmax (V is NOT rotated by RoPE)
    # For K: if RoPE patched, the patch handles it via _current_layer_idx.
    # If not patched, hook k_proj as fallback.
    for name, module in model.named_modules():
        if name.endswith(k_suffix) and not rope_patched:
            idx = extract_layer_idx(name)
            if idx is not None:
                def make_k(i):
                    def fn(mod, inp, out):
                        with torch.no_grad():
                            k_absmax[i] = max(
                                k_absmax[i],
                                out.float().abs().max().item())
                    return fn
                hooks.append(module.register_forward_hook(make_k(idx)))

        elif name.endswith(v_suffix):
            idx = extract_layer_idx(name)
            if idx is not None:
                def make_v(i):
                    def fn(mod, inp, out):
                        with torch.no_grad():
                            v_absmax[i] = max(
                                v_absmax[i],
                                out.float().abs().max().item())
                    return fn
                hooks.append(module.register_forward_hook(make_v(idx)))


    return k_absmax, v_absmax, hooks, _current_layer_idx


def calibrate_sequential(lm, all_input_ids, k_absmax, v_absmax, device,
                         _current_layer_idx=None):
    """Embed → each layer on GPU one at a time → collect K/V absmax."""
    embed = lm.embed_tokens
    layers = lm.layers
    rotary_emb = getattr(lm, "rotary_emb", None)
    num_layers = len(layers)

    # Step 1: embed
    print("  Embedding...")
    embed.to(device)
    all_hidden = []
    with torch.no_grad():
        for i, ids in enumerate(all_input_ids):
            h = embed(ids.unsqueeze(0).to(device))
            all_hidden.append(h.cpu())
            if (i + 1) % 100 == 0:
                print(f"    [{i+1}/{len(all_input_ids)}]")
    embed.cpu()
    torch.cuda.empty_cache()
    print(f"  Embedded {len(all_hidden)} samples")

    # Step 2: layer-by-layer
    if rotary_emb is not None:
        rotary_emb.to(device)

    if _current_layer_idx is None:
        _current_layer_idx = [None]

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        layer.to(device)

        # Set current layer idx for the RoPE hook
        _current_layer_idx[0] = layer_idx

        new_hidden = []
        with torch.no_grad():
            for h in all_hidden:
                h_gpu = h.to(device)
                seq_len = h_gpu.shape[1]
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

                if rotary_emb is not None:
                    cos, sin = rotary_emb(h_gpu, pos_ids)
                    out = layer(h_gpu, position_embeddings=(cos, sin))
                else:
                    try:
                        out = layer(h_gpu, position_ids=pos_ids)
                    except TypeError:
                        out = layer(h_gpu)

                if isinstance(out, tuple):
                    out = out[0]
                new_hidden.append(out.cpu())

        all_hidden = new_hidden
        layer.cpu()
        torch.cuda.empty_cache()

        k_am = k_absmax.get(layer_idx, 0)
        v_am = v_absmax.get(layer_idx, 0)
        tag = f"k={k_am:.4f} v={v_am:.4f}" if k_am > 0 else "(no attention)"
        print(f"  Layer {layer_idx:>3}/{num_layers}: {tag}")

    if rotary_emb is not None:
        rotary_emb.cpu()
    torch.cuda.empty_cache()


def inject_scales(target_dir, k_scales, v_scales):
    """Add k_scale/v_scale tensors to safetensors checkpoint.

    Auto-detects the key prefix by finding an existing self_attn key
    in the checkpoint (e.g. 'model.layers.' vs 'model.language_model.layers.').
    """
    st_files = sorted(f for f in os.listdir(target_dir)
                      if f.endswith(".safetensors"))
    assert st_files, f"No safetensors in {target_dir}"

    index_file = os.path.join(target_dir, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

    # Auto-detect key pattern from existing checkpoint keys.
    # Find a k_proj key for the first attention layer, extract the full
    # path pattern including any .attn. wrapping, skip audio/vision.
    first_attn_layer = min(k_scales.keys())
    from safetensors import safe_open
    k_proj_key = None
    for st_f in st_files:
        with safe_open(os.path.join(target_dir, st_f), framework="pt") as f:
            for key in f.keys():
                if (f"layers.{first_attn_layer}.self_attn" in key
                        and "k_proj" in key
                        and ("weight" in key or "weight_packed" in key)
                        and "audio" not in key and "visual" not in key):
                    k_proj_key = key
                    break
        if k_proj_key:
            break

    if k_proj_key:
        # Extract: everything up to and including "self_attn." + any
        # intermediate modules before "k_proj"
        # e.g. "thinker.model.layers.3.self_attn.k_proj.weight_packed"
        #    → prefix="thinker.model.", attn_infix=""
        # e.g. "model.layers.3.self_attn.attn.k_proj.weight"
        #    → prefix="model.", attn_infix="attn."
        prefix = k_proj_key[:k_proj_key.index(f"layers.{first_attn_layer}.")]
        # Extract everything between "self_attn." and "k_proj"
        sa_idx = k_proj_key.index("self_attn.") + len("self_attn.")
        kp_idx = k_proj_key.index("k_proj")
        attn_infix = k_proj_key[sa_idx:kp_idx]  # e.g. "" or "attn."
    else:
        prefix = "model."
        attn_infix = ""

    print(f"Detected key prefix: '{prefix}', attn_infix: '{attn_infix}'")
    # Scale key pattern: {prefix}layers.{i}.self_attn.{attn_infix}k_scale
    def scale_key(idx, kv):
        return f"{prefix}layers.{idx}.self_attn.{attn_infix}{kv}_scale"

    target_file = st_files[0]
    if weight_map:
        for key, fname in weight_map.items():
            if f"layers.{first_attn_layer}.self_attn" in key:
                target_file = fname
                break

    new_tensors = {}
    for idx in sorted(k_scales.keys()):
        new_tensors[scale_key(idx, "k")] = \
            torch.tensor(k_scales[idx], dtype=torch.float32)
        new_tensors[scale_key(idx, "v")] = \
            torch.tensor(v_scales[idx], dtype=torch.float32)

    print(f"Adding {len(new_tensors)} tensors to {target_file}")

    target_path = os.path.join(target_dir, target_file)
    existing = load_file(target_path)
    existing.update(new_tensors)
    save_file(existing, target_path)

    if weight_map:
        for key in new_tensors:
            weight_map[key] = target_file
        index["weight_map"] = weight_map
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
        print("Updated index")

    # Add kv_cache_scheme to config.json so vLLM loads scales via the
    # standard weight loader (CompressedTensorsKVCacheMethod) instead
    # of needing a custom bypass.
    config_path = os.path.join(target_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        qc = config.get("quantization_config", {})
        qc["kv_cache_scheme"] = {
            "num_bits": 8,
            "type": "int",
            "symmetric": True,
            "strategy": "tensor",
        }
        config["quantization_config"] = qc
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Updated config.json with kv_cache_scheme")


def main():
    args = parse_args()

    print(f"Base model:  {args.base_model}")
    print(f"Target dir:  {args.target_dir}")
    print(f"Model class: {args.model_class or 'AutoModelForCausalLM'}")
    print(f"LM path:     {args.language_model_path or '(standard)'}")
    print(f"Samples:     {args.num_samples}, Max seq len: {args.max_seq_len}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("\nLoading calibration datasets...")
    ds = load_calibration_data(tokenizer, args.num_samples, args.max_seq_len)
    print(f"Loaded {len(ds)} samples")

    print(f"\nLoading {args.base_model} on CPU...")
    model = load_model(args.base_model, args.model_class,
                       quantized=getattr(args, 'quantized', False))
    model.eval()

    lm = resolve_language_model(model, args.language_model_path)

    # Register hooks only on the language model (not audio/vision towers)
    k_absmax, v_absmax, hooks, _current_layer_idx = register_hooks(
        lm, args.k_proj_suffix, args.v_proj_suffix)
    print(f"Hooked {len(hooks)} projections")

    all_input_ids = [
        torch.tensor(ds[i]["input_ids"], dtype=torch.long)
        for i in range(min(len(ds), args.num_samples))
    ]

    print(f"\nCalibrating ({len(all_input_ids)} samples)...")
    calibrate_sequential(lm, all_input_ids, k_absmax, v_absmax, args.device,
                         _current_layer_idx=_current_layer_idx)

    for h in hooks:
        h.remove()

    k_scales, v_scales = {}, {}
    for idx in sorted(k_absmax.keys()):
        if k_absmax[idx] > 0:
            k_scales[idx] = max(k_absmax[idx] / 127.0, 1e-6)
            v_scales[idx] = max(v_absmax[idx] / 127.0, 1e-6)

    print(f"\n{'Layer':>5} {'k_absmax':>10} {'v_absmax':>10} "
          f"{'k_scale':>12} {'v_scale':>12}")
    print("-" * 52)
    for idx in sorted(k_scales.keys()):
        print(f"{idx:>5} {k_absmax[idx]:>10.4f} {v_absmax[idx]:>10.4f} "
              f"{k_scales[idx]:>12.6f} {v_scales[idx]:>12.6f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nInjecting scales into {args.target_dir}...")
    inject_scales(args.target_dir, k_scales, v_scales)

    print(f"\nDone! {len(k_scales)} layers calibrated.")
    print(f"Use: vllm serve {args.target_dir} --kv-cache-dtype int8_per_tensor")


if __name__ == "__main__":
    main()
