# SlimGPT Server

SlimGPT Server is a high-performance, lightweight Flask inference API built for **SlimGPT** — a custom from-scratch ByteLevelBPE Transformer language model. Designed for edge and CPU-friendly cloud deployments (such as Render via Docker), SlimGPT Server features thread-safe lazy model loading, CPU thread optimization, synchronous generation, and real-time Server-Sent Events (SSE) streaming token generation.

**Live Application**: [https://slim-gpt.vercel.app/](https://slim-gpt.vercel.app/)  
**Frontend Repository**: [https://github.com/Pro1943/SlimGPT](https://github.com/Pro1943/SlimGPT)  
**Backend Repository**: [https://github.com/Pro1943/SlimGPT-server](https://github.com/Pro1943/SlimGPT-server)  
**Hugging Face Weights Repo**: [Pro1943/SlimGPT-deploy](https://huggingface.co/Pro1943/SlimGPT-deploy)

---

## Author & Credits

Designed and developed by **Pro 1943**, part of the **GAD (Gen AI Devs)** and **R&D Team of Gen AI Devs**.

---

## Key Features

- **Custom From-Scratch GPT Architecture (`MicroGPT`)**: Pure PyTorch implementation with weight-tied token embeddings and output linear head, causal self-attention via `scaled_dot_product_attention`, GELU feedforward blocks, and pre-LayerNorm layer structure.
- **Server-Sent Events (SSE) Streaming (`POST /generate/stream`)**: Real-time token-by-token streaming delivering low latency live responses to frontend interfaces.
- **Synchronous Generation API (`POST /generate`)**: Standard JSON endpoint returning complete generated text with configurable top-k, top-p, and repetition penalties.
- **Automatic HF Model Ingestion**: Downloads model checkpoint (`model.pkl`) and tokenizer (`tokenizer.json`) automatically from Hugging Face Hub on first request startup.
- **CPU Optimization & Low Memory Footprint**: Configured with `torch.set_num_threads(1)` to eliminate core thrashing on constrained shared cloud CPUs (e.g. Render free/starter tiers), running in FP32 precision for fast native CPU matrix math.
- **Thread-Safe & Production-Ready**: Powered by Gunicorn (`gthread` workers) with `ThreadPoolExecutor` isolation and thread-safe queue streaming.
- **Docker & Render Ready**: Containerized using `python:3.11-slim` with a dedicated CPU-only PyTorch wheel build and Render blueprint (`render.yaml`).

---

## Model Architecture (`MicroGPT`)

SlimGPT is trained using a compact decoder-only transformer architecture matching these exact specifications:

| Parameter | Value |
| :--- | :--- |
| **Vocab Size** | `32771` (ByteLevelBPE with `<|user|>`, `<|assistant|>`, `<|eos|>`) |
| **Embedding Dimension (`embed_dim`)** | `448` |
| **Attention Heads (`num_heads`)** | `7` |
| **Head Dimension (`head_dim`)** | `64` |
| **Transformer Layers (`num_layers`)** | `8` |
| **Context Window (`block_size`)** | `384` |
| **Feedforward Expansion** | `4x` (`1792` dim) with `GELU` activation |
| **Weights Tie** | Token embedding weight tied to output linear head weight |
| **Attention Kernel** | `F.scaled_dot_product_attention(is_causal=True)` |

---

## API Documentation

### 1. Health Check
Checks server status and verifies whether the model has been loaded into memory.

- **Endpoint**: `GET /health`
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 2. Generate (Synchronous)
Generates text completion for a given prompt and returns the full response in a single JSON payload.

- **Endpoint**: `POST /generate`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "max_tokens": 150,
  "temperature": 1.0
}
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `string` | **Required** | The input prompt text. |
| `max_tokens` | `integer` | `150` | Maximum number of new tokens to generate. |
| `temperature` | `float` | `1.0` | Sampling temperature (>0 for probabilistic, 0 for greedy). |

- **Response `200 OK`**:
```json
{
  "response": "Quantum computing is a type of computation that uses quantum mechanics..."
}
```

- **Error Response `504 Gateway Timeout`**: Returned if generation exceeds the 180-second timeout.
```json
{
  "error": "generation timed out after 180s"
}
```

---

### 3. Generate Stream (Real-Time SSE)
Streams token generation live over Server-Sent Events (SSE).

- **Endpoint**: `POST /generate/stream`
- **Headers**: `Content-Type: application/json`
- **Response `Content-Type`**: `text/event-stream`
- **Request Body**:
```json
{
  "prompt": "Write a short poem about space.",
  "max_tokens": 100,
  "temperature": 0.8
}
```

#### Event Format

1. **Per-Token Event (`type: "token"`)**: Emitted after every single token is generated.
```text
data: {"type": "token", "index": 0, "elapsed": 0.0412, "text_so_far": "Stars"}

data: {"type": "token", "index": 1, "elapsed": 0.0835, "text_so_far": "Stars shine"}
```

2. **Completion Event (`type: "done"`)**: Emitted when generation finishes (due to `max_tokens` or `<|eos|>`).
```text
data: {"type": "done", "total_tokens": 42, "elapsed": 1.8451, "response": "Stars shine in the silent night..."}
```

3. **Error Event (`type: "error"`)**: Emitted if generation fails or times out.
```text
data: {"type": "error", "message": "generation timed out after 180s"}
```

---

## Usage Examples

### cURL (Synchronous)
```bash
curl -X POST https://slimgpt-api.onrender.com/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50, "temperature": 0.7}'
```

### cURL (SSE Stream)
```bash
curl -N -X POST https://slimgpt-api.onrender.com/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a joke.", "max_tokens": 60}'
```

### JavaScript / Fetch (Browser EventSource / Stream Reader)
```javascript
async function streamCompletion(prompt) {
  const response = await fetch("https://slimgpt-api.onrender.com/generate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_tokens: 100, temperature: 0.9 })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split("\n\n");

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = JSON.parse(line.replace("data: ", ""));
        if (data.type === "token") {
          console.log("Stream text:", data.text_so_far);
        } else if (data.type === "done") {
          console.log("Final Response:", data.response);
        }
      }
    }
  }
}
```

### Python (`requests` SSE Stream)
```python
import json
import requests

url = "https://slimgpt-api.onrender.com/generate/stream"
payload = {"prompt": "What is AI?", "max_tokens": 80}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            event = json.loads(decoded[6:])
            if event["type"] == "token":
                print(f"\r{event['text_so_far']}", end="")
            elif event["type"] == "done":
                print(f"\nDone in {event['elapsed']}s")
```

---

## Project Structure

```
SlimGPT-server/
├── app.py              # Main Flask application (MicroGPT model, HF loader, SSE & REST endpoints)
├── Dockerfile          # Multi-stage CPU PyTorch Docker container definition
├── requirements.txt    # Production Python dependencies (Flask, Gunicorn, Tokenizers, etc.)
├── render.yaml         # Render Infrastructure-as-Code blueprint
├── .gitignore          # Environment and cache ignore rules
└── README.md           # Documentation
```

---

## Local Setup & Development

### Prerequisites
- Python 3.11+
- Git

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Pro1943/SlimGPT-server.git
cd SlimGPT-server
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

4. **Run the local development server**:
```bash
python app.py
```
*The app will start locally on `http://localhost:5000` (or `http://0.0.0.0:10000`).*

---

## Docker & Render Deployment

### Docker Local Build & Run

```bash
docker build -t slimgpt-server .
docker run -p 10000:10000 -e PORT=10000 slimgpt-server
```

### Deploying to Render

1. Fork or push this repository to GitHub/GitLab.
2. In the Render Dashboard, click **New +** -> **Web Service**.
3. Select your repository. Render will automatically detect the `Dockerfile` and `render.yaml`.
4. *(Optional)* Add environment variable `HF_TOKEN` if using a private Hugging Face model repository.
5. Deploy!

---

## Related Repositories

- **Live Site**: [slim-gpt.vercel.app](https://slim-gpt.vercel.app/)
- **Web Frontend**: [Pro1943/SlimGPT](https://github.com/Pro1943/SlimGPT)
- **Model Server**: [Pro1943/SlimGPT-server](https://github.com/Pro1943/SlimGPT-server)
- **Hugging Face Model**: [Pro1943/SlimGPT-deploy](https://huggingface.co/Pro1943/SlimGPT-deploy)

---

## License

Created by **Pro 1943** & **Gen AI Devs (GAD)**. All rights reserved.
