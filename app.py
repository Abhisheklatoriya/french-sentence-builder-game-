import csv
import io
import json
import random
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).parent
CONTENT_PATH = APP_DIR / "content_bank.json"
PROGRESS_PATH = APP_DIR / "progress.json"


# -------------------------
# Storage helpers
# -------------------------
def save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, fallback: dict | None = None):
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


# -------------------------
# Content validation + parsing
# -------------------------
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def validate_bank(bank: dict) -> tuple[bool, str]:
    if not isinstance(bank, dict):
        return False, "Top-level JSON must be an object like { 'A1': [ ... ], 'A2': [ ... ] }."

    for lvl in LEVELS:
        if lvl not in bank:
            return False, f"Missing level: {lvl}"
        if not isinstance(bank[lvl], list):
            return False, f"Level {lvl} must be a list of items."

        if len(bank[lvl]) < 1:
            return False, f"Level {lvl} has no items."

        for i, item in enumerate(bank[lvl]):
            if not isinstance(item, dict):
                return False, f"{lvl}[{i}] must be an object."
            if "en" not in item or "fr" not in item or "hints" not in item:
                return False, f"{lvl}[{i}] must have keys: en, fr, hints"
            if not isinstance(item["hints"], list) or len(item["hints"]) < 1:
                return False, f"{lvl}[{i}] hints must be a non-empty list."
    return True, "OK"


def parse_csv_content(csv_bytes: bytes) -> dict:
    """
    CSV columns:
    level,en,fr,hint1,hint2,hint3,hint4,... (any number of hint columns)
    """
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    required = {"level", "en", "fr"}
    if not required.issubset(set(reader.fieldnames or [])):
        raise ValueError(f"CSV must include columns: {', '.join(sorted(required))}")

    bank = {lvl: [] for lvl in LEVELS}

    hint_cols = [c for c in (reader.fieldnames or []) if c.lower().startswith("hint")]

    for row in reader:
        lvl = (row.get("level") or "").strip()
        if lvl not in bank:
            raise ValueError(f"Invalid level in CSV: {lvl}. Must be one of {LEVELS}")

        en = (row.get("en") or "").strip()
        fr = (row.get("fr") or "").strip()
        if not en or not fr:
            continue

        hints = []
        for hc in hint_cols:
            v = (row.get(hc) or "").strip()
            if v:
                hints.append(v)
        if not hints:
            hints = ["No hints provided for this item."]

        bank[lvl].append({"en": en, "fr": fr, "hints": hints})

    # ensure every level exists (even if empty)
    return bank


# -------------------------
# Game helpers
# -------------------------
def tokenize(fr: str):
    # Keep punctuation as separate tokens; keep apostrophes inside tokens.
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


def format_tokens(tokens):
    out = ""
    for t in tokens:
        if t in {".", ",", "…"}:
            out = out.rstrip() + t + " "
        elif t in {"?", "!", ":", ";"}:
            out = out.rstrip() + " " + t + " "
        else:
            out += t + " "
    return out.strip()


def new_round(bank, level):
    item = random.choice(bank[level])
    fr_tokens = tokenize(item["fr"])
    shuffled = fr_tokens[:]
    random.shuffle(shuffled)
    return {
        "level": level,
        "en": item["en"],
        "fr": item["fr"],
        "fr_tokens": fr_tokens,
        "shuffled": shuffled,
        "hints": item["hints"],
        "hint_idx": 0,
        "built": [],
    }


def init_state():
    if "bank" not in st.session_state:
        if CONTENT_PATH.exists():
            st.session_state.bank = load_json(CONTENT_PATH, {})
        else:
            st.session_state.bank = {lvl: [] for lvl in LEVELS}

    # If bank file is missing or broken, fail loudly with guidance
    ok, msg = validate_bank(st.session_state.bank) if st.session_state.bank else (False, "No content loaded.")
    if not ok:
        # If you just created the repo, content_bank.json should be valid.
        # This is here to keep the app from silently crashing when content is wrong.
        st.session_state.bank = load_json(CONTENT_PATH, {})
        ok2, msg2 = validate_bank(st.session_state.bank) if st.session_state.bank else (False, "No content loaded.")
        if not ok2:
            st.error("Your content_bank.json is missing or invalid.")
            st.code(msg2)
            st.stop()

    if "progress" not in st.session_state:
        st.session_state.progress = load_json(PROGRESS_PATH, {
            "xp": 0,
            "correct": 0,
            "attempts": 0,
            "by_level": {lvl: {"correct": 0, "attempts": 0} for lvl in LEVELS},
        }) or {
            "xp": 0,
            "correct": 0,
            "attempts": 0,
            "by_level": {lvl: {"correct": 0, "attempts": 0} for lvl in LEVELS},
        }

    if "level" not in st.session_state:
        st.session_state.level = "A1"
    if "round" not in st.session_state:
        st.session_state.round = new_round(st.session_state.bank, st.session_state.level)


def persist_progress():
    try:
        save_json(PROGRESS_PATH, st.session_state.progress)
    except Exception:
        # On Streamlit Community Cloud, writing files is usually OK within the app container,
        # but it may reset on redeploy. That’s fine for a prototype.
        pass


def set_level(level):
    st.session_state.level = level
    st.session_state.round = new_round(st.session_state.bank, level)


def clear_built():
    st.session_state.round["built"] = []
    st.session_state.round["hint_idx"] = 0


def add_token(tok):
    st.session_state.round["built"].append(tok)


def undo_token():
    if st.session_state.round["built"]:
        st.session_state.round["built"].pop()


def check_answer():
    r = st.session_state.round
    p = st.session_state.progress

    p["attempts"] += 1
    p["by_level"][r["level"]]["attempts"] += 1

    correct = r["built"] == r["fr_tokens"]
    if correct:
        p["correct"] += 1
        p["by_level"][r["level"]]["correct"] += 1

        used_hints = r["hint_idx"]
        gained = max(2, 10 - 2 * used_hints)
        p["xp"] += gained
        persist_progress()
        return True, f"✅ Correct! +{gained} XP"
    else:
        persist_progress()
        first_wrong = None
        for i, t in enumerate(r["built"]):
            if i >= len(r["fr_tokens"]) or t != r["fr_tokens"][i]:
                first_wrong = i
                break
        if first_wrong is None and len(r["built"]) != len(r["fr_tokens"]):
            first_wrong = len(r["built"])
        msg = "❌ Not quite."
        if first_wrong is not None:
            msg += f" First mismatch at position {first_wrong + 1}."
        return False, msg


def next_hint():
    r = st.session_state.round
    if r["hint_idx"] < len(r["hints"]) - 1:
        r["hint_idx"] += 1


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="French Sentence Builder", layout="wide")
init_state()

st.title("🇫🇷 French Sentence Builder — Prototype")
st.caption("Prototype: click-to-build sentence order (fast to validate the learning loop). Upload new content as JSON/CSV from the sidebar.")

# Sidebar: levels + admin content upload
with st.sidebar:
    st.header("Game")
    level = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.level))
    if level != st.session_state.level:
        set_level(level)

    st.divider()
    st.header("Progress")
    p = st.session_state.progress
    st.metric("XP", p["xp"])
    st.metric("Correct / Attempts", f"{p['correct']} / {p['attempts']}")
    for lvl in LEVELS:
        bl = p["by_level"][lvl]
        st.write(f"- {lvl}: {bl['correct']} / {bl['attempts']}")

    if st.button("Reset progress"):
        st.session_state.progress = {
            "xp": 0, "correct": 0, "attempts": 0,
            "by_level": {lvl: {"correct": 0, "attempts": 0} for lvl in LEVELS},
        }
        persist_progress()
        st.success("Progress reset.")

    st.divider()
    st.header("Content admin (upload)")
    st.write("Upload **JSON** or **CSV** to replace the entire content bank.")

    up = st.file_uploader("Upload content", type=["json", "csv"])
    if up is not None:
        try:
            if up.name.lower().endswith(".json"):
                bank = json.loads(up.getvalue().decode("utf-8"))
            else:
                bank = parse_csv_content(up.getvalue())

            ok, msg = validate_bank(bank)
            if not ok:
                st.error("Upload failed: invalid content format.")
                st.code(msg)
            else:
                st.session_state.bank = bank
                save_json(CONTENT_PATH, bank)  # persist in repo container
                st.success("✅ Content updated! Starting a fresh round.")
                st.session_state.round = new_round(st.session_state.bank, st.session_state.level)
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.download_button(
        "Download current content_bank.json",
        data=json.dumps(st.session_state.bank, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="content_bank.json",
        mime="application/json",
    )

# Main UI
r = st.session_state.round
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.subheader(f"Round — Level {r['level']}")
    st.markdown(f"**English:** {r['en']}")

    st.write("**Your built French sentence:**")
    built_str = format_tokens(r["built"]) if r["built"] else "_(empty)_"
    st.code(built_str, language="text")

    b1, b2, b3, b4 = st.columns([0.2, 0.2, 0.2, 0.4])
    with b1:
        if st.button("Undo"):
            undo_token()
    with b2:
        if st.button("Clear"):
            clear_built()
    with b3:
        if st.button("Hint"):
            next_hint()
    with b4:
        submitted = st.button("Check ✅", type="primary")

    st.write("**Hint:**")
    st.info(r["hints"][r["hint_idx"]])

    if submitted:
        ok, msg = check_answer()
        if ok:
            st.success(msg)
            with st.expander("Show answer"):
                st.write(r["fr"])
            if st.button("Next round ▶"):
                st.session_state.round = new_round(st.session_state.bank, st.session_state.level)
        else:
            st.error(msg)

with col2:
    st.subheader("Word bank (click to add)")
    st.caption("You must place punctuation tokens too (., ?, etc.).")

    cols = st.columns(4)
    for i, tok in enumerate(r["shuffled"]):
        with cols[i % 4]:
            if st.button(tok, key=f"tok_{i}_{tok}"):
                add_token(tok)

    st.divider()
    if st.button("Shuffle word bank 🔀"):
        random.shuffle(r["shuffled"])
    if st.button("New round (same level) 🎲"):
        st.session_state.round = new_round(st.session_state.bank, st.session_state.level)

st.divider()
st.caption("This is intentionally simple. Next upgrade: accept multiple valid variants + a review/mistakes mode.")
