import os
import json

HISTORY_FILE = "chat_history.json"

def save_chat(question, answer):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append((question, answer))
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f)

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []
