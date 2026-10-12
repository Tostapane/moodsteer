"""Steps 5-6: load model, add steering vectors to activations, generate.  Owner: P1.
Vector file (artifacts/vectors/<axis>.pt): {"vec": Tensor[d_model], "layer": int}. Strength: config.STRENGTH."""
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import config


class FakeModel:
    """No torch, no GPU: canned replies and random activations. For laptop dev and tests."""
    layers = [None] * 4

    def __init__(self):
        self.alphas = {}

    def set_alphas(self, alphas):
        self.alphas = alphas

    def generate(self, messages, max_new_tokens=None, on_token=None):
        tag = ", ".join(f"{k}={v:.2f}" for k, v in self.alphas.items()) or "none"
        reply = f"[fake | steering: {tag}] you said: {messages[-1]['content']}"
        for word in reply.split(" ") if on_token else []:
            time.sleep(0.02)
            on_token(word + " ")
        return reply

    def generate_batch(self, convos, max_new_tokens=None):
        return [self.generate(c) for c in convos]

    def acts(self, texts, layer=None, starts=None):
        return [[random.random() for _ in range(8)] for _ in texts]


def quiet():
    """Silence Hugging Face progress bars, download notices and warnings."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import huggingface_hub
    import transformers

    huggingface_hub.utils.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()


class Model:
    def __init__(self):
        quiet()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
        self.tok.padding_side = "left"  # batched generation
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, dtype=getattr(torch, config.DTYPE),
                                                          device_map="auto")
        self.layers = self.model.model.layers
        self.vectors = {p.stem: torch.load(p) for p in sorted(Path(config.VECTORS_DIR).glob("*.pt"))}
        self.steer = {}  # layer index → tensor added to that layer's output
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))

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
        """alphas are -1..1 per axis (sign picks the pole), scaled by config.STRENGTH. Unknown axes are skipped."""
        self.steer = {}
        for name, a in alphas.items():
            v = self.vectors.get(name)
            if v is None or a == 0:
                continue
            add = a * config.STRENGTH.get(name, 4.0) * v["vec"]
            self.steer[v["layer"]] = self.steer.get(v["layer"], 0) + add

    @contextmanager
    def unsteered(self):
        steer, self.steer = self.steer, {}
        try:
            yield
        finally:
            self.steer = steer

    def chat_text(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def generate(self, messages, max_new_tokens=None, on_token=None):
        """on_token(text) is called with each new piece of text as it is generated."""
        if on_token is None:
            return self.generate_batch([messages], max_new_tokens)[0]
        from threading import Thread

        from transformers import TextIteratorStreamer

        enc = self.tok(self.chat_text(messages), return_tensors="pt", add_special_tokens=False).to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self.model.generate, kwargs=dict(
            **enc, streamer=streamer, max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=self.tok.pad_token_id))
        thread.start()
        reply = ""
        for text in streamer:
            reply += text
            on_token(text)
        thread.join()
        return reply.strip()

    def generate_batch(self, convos, max_new_tokens=None):
        enc = self.tok([self.chat_text(c) for c in convos], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in out]

    def acts(self, texts, layer=None, starts=None):
        """Unsteered hidden state at `layer`, averaged over tokens[start:] of each text → Tensor[n, d_model].
        Default: middle layer, all tokens."""
        layer = len(self.layers) // 2 if layer is None else layer
        out = []
        with self.unsteered(), self.torch.no_grad():
            for text, start in zip(texts, starts or [0] * len(texts)):
                enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                hs = self.model(**enc, output_hidden_states=True).hidden_states[layer + 1]  # [0] = embeddings
                out.append(hs[0, start:].float().mean(0).cpu())
        return self.torch.stack(out)


_model = None


def get_model():
    global _model
    if _model is None:
        _model = FakeModel() if config.FAKE else Model()
    return _model
