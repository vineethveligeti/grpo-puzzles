#!/usr/bin/env python
"""Write a text-only copy of a Qwen3.5 checkpoint that vLLM loads as `Qwen3_5ForCausalLM`.

    python make_text_only_ckpt.py Qwen/Qwen3.5-0.8B-Base /content/qwen3.5-0.8b-text

Why (verified 2026-09-08, vllm==0.27.1, transformers==5.16.1, trl==1.12.0):
  * The Hub checkpoint is a vision-language model (`Qwen3_5ForConditionalGeneration`). vLLM instantiates that class,
    but TRL's colocated weight sync sends the *text-only* trainer's parameter names (`model.layers...`) ->
    "There is no module or parameter named 'model' in Qwen3_5ForConditionalGeneration"  (TRL #5269, still open).
  * vLLM 0.27.1 does register a text-only `Qwen3_5ForCausalLM`, and picks it when config.json says so.
  * `save_pretrained` of the text-only HF model keeps the *original* tensor names (`model.language_model.layers...`),
    which the text-only vLLM class rejects, so the safetensors keys are rewritten here too. Vision + MTP tensors are dropped.
Point BOTH `--model` (trainer) and therefore vLLM at the output dir.
"""
import glob, os, sys, torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
if not os.path.exists(os.path.join(out, "config.json")):
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)   # -> Qwen3_5ForCausalLM (text only)
    m.save_pretrained(out); AutoTokenizer.from_pretrained(src).save_pretrained(out)
    print("saved", type(m).__name__, "->", out)

files = sorted(glob.glob(os.path.join(out, "*.safetensors"))); sd = {}
for f in files: sd.update(load_file(f))
if any(k.startswith("model.language_model.") for k in sd):
    new = {}
    for k, v in sd.items():
        if k.startswith("model.language_model."): new["model." + k[len("model.language_model."):]] = v
        elif k.startswith("model.visual") or k.startswith("mtp."): continue
        else: new[k] = v
    for f in files: os.remove(f)
    idx = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(idx): os.remove(idx)
    save_file(new, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    ref = sd["model.language_model.layers.3.mlp.down_proj.weight"]
    print(f"rewrote {len(sd)} -> {len(new)} tensors")
else:
    ref = sd["model.layers.3.mlp.down_proj.weight"]; print("already text-only")

m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
w = dict(m.named_parameters())["model.layers.3.mlp.down_proj.weight"]
assert torch.equal(w.cpu(), ref), "reload mismatch"
print("verified:", type(m).__name__, f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M params, weights identical")
