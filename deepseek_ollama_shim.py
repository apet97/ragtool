#!/usr/bin/env python3
import os, json, time, ssl, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import URLError

API_BASE = os.environ.get("DEEPSEEK_API_BASE","https://api.deepseek.com/v1")
API_KEY  = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL    = os.environ.get("DEEPSEEK_MODEL","deepseek-chat")
HOST     = os.environ.get("SHIM_HOST","127.0.0.1")
PORT     = int(os.environ.get("SHIM_PORT","11434"))

if not API_KEY:
    print("ERROR: set DEEPSEEK_API_KEY", file=sys.stderr)
    sys.exit(1)

print(f"[shim] Starting: http://{HOST}:{PORT}")
print(f"[shim] Chat: {API_BASE}/chat/completions (model={MODEL})")
print(f"[shim] Embeddings: Local (sentence-transformers)")
sys.stdout.flush()

# Load embeddings model
try:
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[shim] ✅ Embeddings model loaded")
except Exception as e:
    print(f"[shim] ⚠️  Embeddings not available: {e}", file=sys.stderr)
    emb_model = None

def ds_chat(messages, temperature=0.0):
    """Call DeepSeek API"""
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "max_tokens": 4096
        }).encode()
        
        req = Request(
            f"{API_BASE}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
        )
        
        ctx = ssl.create_default_context()
        with urlopen(req, context=ctx, timeout=30) as r:
            data = json.loads(r.read())
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return ""
    except Exception as e:
        print(f"[API ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise

def get_embeddings(text_list):
    """Get embeddings locally"""
    if emb_model is None:
        raise RuntimeError("Embeddings model not available")
    
    embeddings = emb_model.encode(text_list, convert_to_numpy=True)
    return embeddings

class H(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    
    def _ok(self, ct="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.end_headers()
    
    def _error(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": msg}).encode())
    
    def do_GET(self):
        if self.path == "/api/tags":
            self._ok()
            self.wfile.write(json.dumps({
                "models": [
                    {"name": "deepseek-chat:latest"},
                    {"name": "nomic-embed-text:latest"}
                ]
            }).encode())
        else:
            self._error(404, "Not found")
    
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw)
        except Exception as e:
            self._error(400, f"Bad request: {e}")
            return
        
        try:
            if self.path == "/api/generate":
                messages = payload.get("messages") or [{"role": "user", "content": payload.get("prompt", "")}]
                temperature = float(payload.get("options", {}).get("temperature", 0.0))
                text = ds_chat(messages, temperature)
                
                self._ok()
                response = {
                    "model": MODEL,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "response": text,
                    "done": True
                }
                self.wfile.write(json.dumps(response).encode())
                return
            
            elif self.path == "/api/chat":
                messages = payload.get("messages", [])
                temperature = float(payload.get("options", {}).get("temperature", 0.0))
                text = ds_chat(messages, temperature)
                
                self._ok()
                response = {
                    "model": MODEL,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": text},
                    "done": True
                }
                self.wfile.write(json.dumps(response).encode())
                return
            
            elif self.path == "/api/embeddings":
                # Handle both Ollama and OpenAI formats
                input_data = payload.get("input")
                
                if isinstance(input_data, str):
                    texts = [input_data]
                elif isinstance(input_data, list):
                    texts = input_data
                else:
                    texts = [str(input_data)]
                
                embeddings = get_embeddings(texts)
                
                # Return in Ollama format
                self._ok()
                response = {
                    "model": payload.get("model", "nomic-embed-text"),
                    "embedding": embeddings[0].tolist() if len(embeddings) == 1 else embeddings.tolist(),
                    "embeddings": [e.tolist() for e in embeddings]
                }
                self.wfile.write(json.dumps(response).encode())
                return
            
            else:
                self._ok()
                self.wfile.write(b"{}")
        
        except Exception as e:
            self._error(500, f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    try:
        server = HTTPServer((HOST, PORT), H)
        print(f"[shim] ✅ Listening on {HOST}:{PORT}", file=sys.stderr)
        sys.stderr.flush()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shim] Stopped")
        sys.exit(0)

