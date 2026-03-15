import json
import random
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

APP_DIR = Path(__file__).parent
CONTENT_PATH = APP_DIR / "content_bank.json"
PROGRESS_PATH = APP_DIR / "progress.json"
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def load_bank():
    return json.loads(CONTENT_PATH.read_text(encoding="utf-8"))


def tokenize(fr: str):
    raw = (
        fr.replace("?", " ?")
        .replace("!", " !")
        .replace(".", " .")
        .replace(",", " ,")
        .replace(":", " :")
        .replace(";", " ;")
        .replace("…", " …")
    )
    return raw.split()


@app.route("/")
def index():
    return render_template("index.html", levels=LEVELS)


@app.route("/api/round")
def get_round():
    level = request.args.get("level", "A1")
    bank = load_bank()
    if level not in bank or not bank[level]:
        return jsonify({"error": f"No content for level {level}"}), 400

    item = random.choice(bank[level])
    fr_tokens = tokenize(item["fr"])
    word_types = item.get("word_types")

    indexed = [
        {"tok": tok, "type": (word_types[i] if word_types else None), "orig_idx": i}
        for i, tok in enumerate(fr_tokens)
    ]
    shuffled = indexed[:]
    random.shuffle(shuffled)

    return jsonify({
        "level": level,
        "en": item["en"],
        "fr": item["fr"],
        "fr_tokens": fr_tokens,
        "has_buckets": word_types is not None,
        "shuffled": shuffled,
        "hints": item["hints"],
    })


@app.route("/api/check", methods=["POST"])
def check_answer():
    data = request.json
    built = data.get("built", [])
    fr_tokens = data.get("fr_tokens", [])
    correct = built == fr_tokens

    first_wrong = None
    if not correct:
        for i, t in enumerate(built):
            if i >= len(fr_tokens) or t != fr_tokens[i]:
                first_wrong = i
                break
        if first_wrong is None and len(built) != len(fr_tokens):
            first_wrong = len(built)

    return jsonify({"correct": correct, "first_wrong": first_wrong})


@app.route("/api/progress", methods=["GET"])
def get_progress():
    default = {
        "xp": 0, "correct": 0, "attempts": 0,
        "by_level": {lvl: {"correct": 0, "attempts": 0} for lvl in LEVELS},
    }
    if PROGRESS_PATH.exists():
        try:
            return jsonify(json.loads(PROGRESS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return jsonify(default)


@app.route("/api/progress", methods=["POST"])
def save_progress():
    data = request.json
    try:
        PROGRESS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
