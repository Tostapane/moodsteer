"""HTTP API. Only this process touches the GPU.  Owner: P1.
  POST /chat   {"session_id", "msg", "steer", "compare"} → {"reply", "scores", "mood", "alphas", ["baseline"]}
  POST /reset  {"session_id"}
  POST /acts   {"texts": [...], "layer": int|null} → {"acts": [[...], ...]}  (mean hidden state per text)
  GET  /health
Run: python -m core.server   (FAKE=1 python -m core.server for no-GPU dev)"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from core.chat import chat, reset
from core.model import get_model


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "model": config.MODEL_ID, "fake": config.FAKE})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/chat":
                return self._send(200, chat(body["session_id"], body["msg"], body.get("steer", True),
                                             body.get("compare", False)))
            if self.path == "/reset":
                reset(body["session_id"])
                return self._send(200, {"ok": True})
            if self.path == "/acts":
                h = get_model().acts(body["texts"], body.get("layer"))
                return self._send(200, {"acts": h.tolist() if hasattr(h, "tolist") else h})
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": repr(e)})


if __name__ == "__main__":
    get_model()  # load before accepting requests
    print(f"serving on :{config.PORT} (fake={config.FAKE})")
    ThreadingHTTPServer(("0.0.0.0", config.PORT), Handler).serve_forever()
