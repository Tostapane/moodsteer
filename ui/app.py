"""Demo UI: same messages to steering ON and OFF, side by side, with live mood + alpha bars.  Owner: P4.
Run: python ui/app.py   (SERVER=http://gpu-box:8000 to point elsewhere)"""
import json
import os
import urllib.request
import uuid

import gradio as gr

SERVER = os.getenv("SERVER", "http://localhost:8000")


def post(path, body):
    req = urllib.request.Request(SERVER + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def send(msg, on_hist, off_hist, sid):
    sid = sid or uuid.uuid4().hex[:8]
    on = post("/chat", {"session_id": sid + "-on", "msg": msg, "steer": True})
    off = post("/chat", {"session_id": sid + "-off", "msg": msg, "steer": False})
    user = {"role": "user", "content": msg}
    on_hist = on_hist + [user, {"role": "assistant", "content": on["reply"]}]
    off_hist = off_hist + [user, {"role": "assistant", "content": off["reply"]}]
    return "", on_hist, off_hist, sid, on["mood"], on["alphas"] or {"(no steering)": 0.0}


with gr.Blocks(title="Moodsteer") as demo:
    sid = gr.State("")
    with gr.Row():
        on = gr.Chatbot(label="Steering ON", type="messages", height=450)
        off = gr.Chatbot(label="Steering OFF", type="messages", height=450)
    with gr.Row():
        mood = gr.Label(label="Mood (smoothed)", num_top_classes=9)
        alphas = gr.Label(label="Steering α")
    box = gr.Textbox(placeholder="Type a message…", show_label=False)
    new = gr.Button("New chat")
    box.submit(send, [box, on, off, sid], [box, on, off, sid, mood, alphas])
    new.click(lambda: ([], [], "", None, None), None, [on, off, sid, mood, alphas])

if __name__ == "__main__":
    demo.launch()
