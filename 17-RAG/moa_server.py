"""모아 AI 비서 서버 (Flask)  —  실행: python moa_server.py  →  http://localhost:5001"""
import base64, io, os, threading
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import chromadb, requests, torch
import torch.nn.functional as F
from chromadb.config import Settings
from flask import Flask, jsonify, request
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parent
WORK = BASE / "work"
EMB_NAME = "intfloat/multilingual-e5-small"
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
TRIGGER = "moa_character"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

app = Flask(__name__)

# ─── 서버가 켜질 때 '한 번만' 모델을 불러 둔다 (요청마다 불러오면 너무 느리니까) ───
print("⏳ 모델 불러오는 중...", flush=True)
tok = AutoTokenizer.from_pretrained(EMB_NAME)
emb_model = AutoModel.from_pretrained(EMB_NAME).eval().to(DEVICE)                        # 귀 (임베딩)
sent_model = AutoModelForSequenceClassification.from_pretrained(WORK / "models" / "moa-sentiment").eval().to(DEVICE)   # 후기 감정 분류기
collection = chromadb.PersistentClient(path=str(WORK / "chroma"), settings=Settings(anonymized_telemetry=False)).get_collection("moa_knowledge")
_sd = {"pipe": None}                                                                      # 그림 모델은 첫 요청 때 불러온다(게으른 로딩)
_sd_lock = threading.Lock()                                                                # 그림은 한 번에 한 명씩

def embed(texts, prefix):
    enc = tok([prefix + t for t in texts], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        h = emb_model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    return F.normalize((h * m).sum(1) / m.sum(1), p=2, dim=1).cpu().tolist()

def ask_llm(messages, temperature=0.1):
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={"model": LLM_MODEL, "messages": messages, "stream": False,
                                                       "options": {"temperature": temperature}, "keep_alive": "30m"}, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]

RAG_SYSTEM = ("너는 대학 AI 입문 동아리 '모아'의 AI 비서 '모아'야. 반드시 아래 [참고 자료]에 있는 내용만 근거로 한국어 존댓말로 간결하게 대답해. "
              "자료에 답이 없으면 지어내지 말고 '제가 가진 자료에는 없는 내용이에요. 임원진에게 문의해 주세요.'라고 말해.")

# ─── 엔드포인트 (손님이 주문하는 창구들) ───
@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "chunks": collection.count()})

@app.post("/sentiment")
def sentiment():
    body = request.get_json(force=True)
    texts = body.get("texts") or [body.get("text", "")]
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(sent_model(**enc).logits, dim=-1).cpu()
    labels = sent_model.config.id2label
    return jsonify([{"text": t, "label": labels[int(p.argmax())], "confidence": round(float(p.max()), 3)} for t, p in zip(texts, probs)])

@app.post("/ask")
def ask():
    question = request.get_json(force=True)["question"]
    res = collection.query(query_embeddings=embed([question], "query: "), n_results=3)
    sources = [m["source"] for m in res["metadatas"][0]]
    context = "\n\n".join(f"({s}) {d}" for s, d in zip(sources, res["documents"][0]))
    answer = ask_llm([{"role": "system", "content": RAG_SYSTEM},
                      {"role": "user", "content": f"[참고 자료]\n{context}\n\n[질문]\n{question}"}])
    return jsonify({"question": question, "answer": answer, "sources": sorted(set(sources))})

@app.post("/draw")
def draw():
    body = request.get_json(force=True)
    prompt, seed, steps = body["prompt"], int(body.get("seed", 0)), int(body.get("steps", 20))
    with _sd_lock:
        if _sd["pipe"] is None:
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
            dtype = torch.float16 if DEVICE != "cpu" else torch.float32
            p = StableDiffusionPipeline.from_pretrained(SD_ID, torch_dtype=dtype, variant="fp16", safety_checker=None,
                                                        requires_safety_checker=False).to(DEVICE)
            p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
            p.set_progress_bar_config(disable=True)
            if (WORK / "models" / "moa-mascot-lora").exists():
                p.load_lora_weights(str(WORK / "models" / "moa-mascot-lora"))                  # 마스코트 LoRA 장착
            _sd["pipe"] = p
        image = _sd["pipe"](prompt, negative_prompt="low quality, blurry, deformed, watermark, text", num_inference_steps=steps,
                            generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    buf = io.BytesIO(); image.save(buf, format="PNG")
    return jsonify({"prompt": prompt, "image_base64": base64.b64encode(buf.getvalue()).decode()})

PAGE = """<!doctype html><meta charset=utf-8><title>모아 AI 비서</title>
<body style="font-family:sans-serif;max-width:680px;margin:30px auto;padding:0 16px">
<h2>🤖 모아 AI 비서</h2>
<h3>💬 동아리에 대해 물어보세요</h3>
<input id=q style="width:75%;padding:8px" placeholder="예: 회비는 누구한테 내요?" onkeydown="if(event.key=='Enter')ask()">
<button onclick="ask()">질문</button><pre id=a style="white-space:pre-wrap;background:#f3f4f6;padding:10px"></pre>
<h3>😊 후기 감정 분석</h3>
<input id=r style="width:75%;padding:8px" placeholder="예: 간식도 맛있고 너무 즐거웠어요">
<button onclick="senti()">분석</button><pre id=s style="background:#f3f4f6;padding:10px"></pre>
<script>
async function post(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}
async function ask(){a.textContent='생각 중...';const d=await post('/ask',{question:q.value});a.textContent=d.answer+'\\n\\n📎 '+d.sources.join(', ')}
async function senti(){const d=await post('/sentiment',{text:r.value});s.textContent=d[0].label+' ('+Math.round(d[0].confidence*100)+'%)'}
</script></body>"""

@app.get("/")
def index():
    return PAGE

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)), threaded=True)
