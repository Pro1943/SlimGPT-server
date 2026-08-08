import os
import sys
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from flask import Flask, request, jsonify
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL = None
TOK = None
DEVICE = torch.device("cpu")
_model_lock = threading.Lock()
_model_loaded = False


def _log(msg):
    print(f"[SlimGPT] {msg}", flush=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SlimGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def _load_model():
    global MODEL, TOK, _model_loaded

    t_total = time.time()
    hf_token = os.environ.get("HF_TOKEN", None)
    repo_id = "Pro1943/SlimGPT-deploy"

    _log("Downloading tokenizer.json from HF Hub...")
    t0 = time.time()
    tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
    _log(f"Tokenizer downloaded in {time.time() - t0:.1f}s")

    _log("Downloading model.pkl from HF Hub...")
    t0 = time.time()
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl", token=hf_token)
    _log(f"model.pkl downloaded in {time.time() - t0:.1f}s")

    _log("Loading tokenizer...")
    t0 = time.time()
    TOK = Tokenizer.from_file(tok_path)
    _log(f"Tokenizer loaded in {time.time() - t0:.1f}s")

    _log("Unpickling checkpoint...")
    t0 = time.time()
    with open(model_path, "rb") as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    _log(f"Checkpoint unpickled in {time.time() - t0:.1f}s")

    cfg = checkpoint["config"]
    vocab_size = TOK.get_vocab_size()

    _log(f"Constructing SlimGPT (vocab={vocab_size}, embed={cfg.get('embed_dim', 448)}, layers={cfg.get('num_layers', 8)})...")
    t0 = time.time()
    model = SlimGPT(
        vocab_size=vocab_size,
        embed_dim=cfg.get("embed_dim", 448),
        num_heads=cfg.get("num_heads", 7),
        num_layers=cfg.get("num_layers", 8),
        block_size=cfg.get("block_size", 384),
        dropout=0.0,
    )
    _log(f"Model constructed in {time.time() - t0:.1f}s")

    _log("Loading state dict...")
    t0 = time.time()
    model.load_state_dict(checkpoint["model_state"])
    _log(f"State dict loaded in {time.time() - t0:.1f}s")

    del checkpoint

    model.eval()
    model.to(DEVICE)

    _log("Casting model to fp16...")
    t0 = time.time()
    model.half()
    _log(f"fp16 cast done in {time.time() - t0:.1f}s")

    MODEL = model
    _model_loaded = True
    _log(f"Model ready. Total load time: {time.time() - t_total:.1f}s")


def ensure_model_loaded():
    global _model_loaded
    if _model_loaded:
        return
    with _model_lock:
        if _model_loaded:
            return
        _load_model()


@torch.no_grad()
def generate(prompt_ids, max_tokens, temperature, top_k, top_p, repetition_penalty, eos_id):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated = []

    for _ in range(max_tokens):
        idx_cond = idx[:, -MODEL.block_size:]
        logits = MODEL(idx_cond)[:, -1, :].float()

        for token_id in set(prompt_ids + generated):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        if temperature > 0:
            logits = logits / temperature
        else:
            next_id = logits.argmax(dim=-1).item()
            if next_id == eos_id:
                break
            generated.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            continue

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_mask = cumulative_probs - probs > top_p
        sorted_logits[sorted_mask] = -float("inf")

        if top_k > 0:
            sorted_logits[:, top_k:] = -float("inf")

        probs = F.softmax(sorted_logits, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        next_id = sorted_indices.gather(-1, sampled_index).item()

        if next_id == eos_id:
            break

        generated.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=DEVICE)], dim=1)

    return generated


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": _model_loaded})


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    ensure_model_loaded()

    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 150)
    temperature = data.get("temperature", 1.0)

    wrapped = f"<|user|>{prompt}<|assistant|>"
    prompt_ids = TOK.encode(wrapped).ids

    eos_id = TOK.token_to_id("<|eos|>")

    generated_ids = generate(
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_id=eos_id,
    )

    response_text = TOK.decode(generated_ids)
    return jsonify({"response": response_text})
