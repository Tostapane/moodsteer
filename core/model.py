"""Steps 5-6: load model, add steering vectors to activations, generate.  Owner: P1.
Vector file format (artifacts/vectors/<behaviour>.pt): {"vec": Tensor[d_model], "layer": int, "max_alpha": float}"""
import random
from pathlib import Path

import config


class FakeModel:
    def __init__(self):
        self.alphas = {}

    def set_alphas(self, alphas):
        self.alphas = alphas

    def generate(self, messages):
        tag = ", ".join(f"{k}={v:.2f}" for k, v in self.alphas.items()) or "none"
        return f"[fake | steering: {tag}] you said: {messages[-1]['content']}"

    def hidden(self, texts, layer=None):
        return [[random.random() for _ in range(8)] for _ in texts]


class Model:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, dtype=torch.bfloat16, device_map="auto")
        self.layers = self.model.model.layers
        self.vectors = {p.stem: torch.load(p) for p in Path(config.VECTORS_DIR).glob("*.pt")}
        self.steer = {}  # layer index → tensor added to that layer's output
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))
        print(f"loaded {config.MODEL_ID} ({len(self.layers)} layers), vectors: {list(self.vectors) or 'none'}")

    def _hook(self, i):
        def hook(module, inp, out):
            v = self.steer.get(i)
            if v is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = h + v.to(h.device, h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    def set_alphas(self, alphas):
        """alphas are 0..1 per behaviour, scaled by each vector's max_alpha. Unknown behaviours are skipped."""
        self.steer = {}
        for name, a in alphas.items():
            v = self.vectors.get(name)
            if v is None or a == 0:
                continue
            self.steer[v["layer"]] = self.steer.get(v["layer"], 0) + a * v["max_alpha"] * v["vec"]

    def generate(self, messages):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def hidden(self, texts, layer=None):
        """Last-token hidden state at `layer` (default: middle), unsteered. For vectors/ and detector/."""
        layer = len(self.layers) // 2 if layer is None else layer
        steer, self.steer = self.steer, {}
        try:
            out = []
            with self.torch.no_grad():
                for t in texts:
                    enc = self.tok(t, return_tensors="pt").to(self.model.device)
                    hs = self.model(**enc, output_hidden_states=True).hidden_states
                    out.append(hs[layer + 1][0, -1].float().cpu())  # hs[0] = embeddings
            return self.torch.stack(out)
        finally:
            self.steer = steer


_model = None


def get_model():
    global _model
    if _model is None:
        _model = FakeModel() if config.FAKE else Model()
    return _model
