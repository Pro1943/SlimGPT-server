import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from flask import Flask, request, jsonify
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL = None
TOK = None
DEVICE = torch.device("cpu")


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


def load_model():
    global MODEL, TOK

    hf_token = os.environ.get("HF_TOKEN", None)
    repo_id = "Pro1943/SlimGPT-deploy"

    tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", token=hf_token)
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl", token=hf_token)

    TOK = Tokenizer.from_file(tok_path)

    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)

    cfg = checkpoint["config"]
    vocab_size = TOK.get_vocab_size()

    model = SlimGPT(
        vocab_size=vocab_size,
        embed_dim=cfg.get("embed_dim", 448),
        num_heads=cfg.get("num_heads", 7),
        num_layers=cfg.get("num_layers", 8),
        block_size=cfg.get("block_size", 384),
        dropout=0.0,
    )

    state_dict = checkpoint["model_state"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    model.load_state_dict(cleaned)
    model.eval()
    model.to(DEVICE)
    MODEL = model


@torch.no_grad()
def generate(prompt_ids, max_tokens, temperature, top_k, top_p, repetition_penalty, eos_id):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated = []

    for _ in range(max_tokens):
        idx_cond = idx[:, -MODEL.block_size:]
        logits = MODEL(idx_cond)[:, -1, :]

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
    return jsonify({"status": "healthy", "model_loaded": MODEL is not None})


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    if MODEL is None or TOK is None:
        return jsonify({"error": "Model not loaded"}), 503

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


load_model()
