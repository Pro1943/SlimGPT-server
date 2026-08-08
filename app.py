import os
import sys
import time
import signal
import threading
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F
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


class GenerationTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise GenerationTimeoutError("Generation request timed out after 90s")


def _log(msg):
    print(f"[SlimGPT] {msg}", flush=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.qkv        = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj       = nn.Linear(embed_dim, embed_dim)
        self.attn_drop  = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ff   = FeedForward(embed_dim, dropout)
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class MicroGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout):
        super().__init__()
        self.tok_emb     = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb     = nn.Embedding(block_size, embed_dim)
        self.drop        = nn.Dropout(dropout)
        self.blocks      = nn.Sequential(*[Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln          = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.block_size  = block_size

    def forward(self, idx):
        B, T   = idx.shape
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        logits = self.head(self.ln(self.blocks(x)))
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2, eos_id=None):
        _log(f"Starting generation (max_new_tokens={max_new_tokens}, prompt_len={idx.size(1)})...")
        t_start = time.time()
        tokens_generated = 0
        for i in range(max_new_tokens):
            _log(f"[gen] token {i} at {time.time() - t_start:.2f}s")
            logits = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :].float() / temperature
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < vals[:, -1:], float('-inf'))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
            tokens_generated += 1
            if eos_id is not None and next_tok.item() == eos_id:
                break
        elapsed = time.time() - t_start
        speed = tokens_generated / elapsed if elapsed > 0 else 0.0
        _log(f"Generation complete: {tokens_generated} tokens in {elapsed:.2f}s ({speed:.2f} tok/s)")
        return idx


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

    cfg = checkpoint.get("config", {})

    _log("Constructing MicroGPT...")
    t0 = time.time()
    model = MicroGPT(
        vocab_size=32771,
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

    _log("Casting model to fp32...")
    t0 = time.time()
    model.float()
    _log(f"fp32 cast done in {time.time() - t0:.1f}s")

    model.eval()
    model.to(DEVICE)

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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": _model_loaded})


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    ensure_model_loaded()

    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 1.0)
    # DEBUG: hardcode max_tokens to 5 for debugging pass
    max_tokens = 5

    _log(f"Received /generate request: prompt={prompt!r}, max_tokens={max_tokens} (debug hardcoded), temp={temperature}")
    t0 = time.time()

    wrapped = f"<|user|>{prompt}<|assistant|>"
    prompt_ids = TOK.encode(wrapped).ids

    eos_id = TOK.token_to_id("<|eos|>")
    _log(f"Calling MODEL.generate with eos_id={eos_id!r} (prompt_len={len(prompt_ids)})...")

    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(90)

    try:
        idx_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
        out_tensor = MODEL.generate(
            idx_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_id=eos_id,
        )
        t1 = time.time()
        num_generated = out_tensor.size(1) - len(prompt_ids)
        _log(f"MODEL.generate returned {num_generated} tokens in {t1 - t0:.2f}s")
    except GenerationTimeoutError:
        _log("Generation timed out after 90s!")
        return jsonify({"error": "Generation timed out after 90 seconds"}), 504
    finally:
        if use_alarm:
            signal.alarm(0)

    new_ids = out_tensor[0, len(prompt_ids):].tolist()
    response_text = TOK.decode(new_ids)
    _log(f"Endpoint total request duration: {time.time() - t0:.2f}s")
    return jsonify({"response": response_text})
