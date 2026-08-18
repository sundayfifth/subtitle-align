"""ตัวช่วยที่ align.py และ make_srt.py ใช้ร่วมกัน"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# โฟลเดอร์งาน: อ้างอิงจากที่ที่รันคำสั่ง ไม่ใช่ที่ที่ script อยู่
# ทำให้ติดตั้งเป็น skill ไว้ที่เดียว แล้วรันจากโฟลเดอร์วิดีโอไหนก็ได้
WORK = Path(os.environ.get("SUBTITLE_WORK", "work"))


def set_work_dir(path: str | None) -> None:
    global WORK
    if path:
        WORK = Path(path).expanduser()


def load_api_key() -> str:
    """หา GEMINI_API_KEY จาก env ก่อน แล้วค่อยไล่หา .env"""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()

    candidates = [
        Path(".env"),                                   # โฟลเดอร์ที่รันอยู่
        ROOT / ".env",                                   # ข้าง ๆ script
        Path.home() / ".config" / "subtitle-align" / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise SystemExit(
        "ไม่พบ GEMINI_API_KEY\n"
        "ใส่ได้ 3 ที่ (ไล่ตามลำดับ):\n"
        "  1. export GEMINI_API_KEY=... ก่อนรัน\n"
        "  2. ไฟล์ .env ในโฟลเดอร์ที่รันอยู่ หรือข้าง ๆ align.py\n"
        "  3. ~/.config/subtitle-align/.env\n"
        "ขอ key ได้ที่ https://aistudio.google.com/apikey"
    )


def job_dir(name: str) -> Path:
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def fmt_ts(seconds: float, sep: str = ",") -> str:
    """1.234 -> 00:00:01,234 (SRT ใช้ comma, ไฟล์ edit ใช้ dot ให้อ่านง่าย)"""
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def load_words(name: str) -> list[dict]:
    path = job_dir(name) / "words.json"
    if not path.exists():
        raise SystemExit(f"ไม่พบ {path} — รัน align.py ก่อน")
    return json.loads(path.read_text())["words"]


def build_char_timeline(words: list[dict]):
    """กาง word-level timestamps ออกเป็นระดับตัวอักษร

    คืน (text, starts, ends) โดย starts[i]/ends[i] คือเวลาของตัวอักษรที่ i
    ภายในหนึ่งคำจะเกลี่ยเวลาแบบเชิงเส้น — จำเป็นสำหรับภาษาไทยที่ Whisper
    ตัด token เป็นชิ้นยาวบ้างสั้นบ้าง ไม่ตรงกับคำที่คนอ่านเห็น
    """
    text_parts, starts, ends = [], [], []
    for w in words:
        t = w["word"]
        if not t:
            continue
        n = len(t)
        dur = max(0.0, w["end"] - w["start"])
        for i, ch in enumerate(t):
            text_parts.append(ch)
            starts.append(w["start"] + dur * (i / n))
            ends.append(w["start"] + dur * ((i + 1) / n))
    return "".join(text_parts), starts, ends


NORM_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """ตัดช่องว่างทิ้งเพื่อเทียบข้อความ — ไทยเว้นวรรคไม่แน่นอน"""
    return NORM_RE.sub("", s)
