#!/usr/bin/env python3
"""ขั้นที่ 1: วิดีโอ -> transcript (Gemini) -> word-level timestamps (stable-ts)

    python3 align.py ../clip15/clip15.mp4

ผลลัพธ์อยู่ใน work/<name>/
    audio.flac      เสียง 16kHz mono ที่ดึงออกมา
    transcript.txt  ข้อความดิบจาก Gemini (แก้ได้ ถ้าถอดผิดคำ แล้วรันซ้ำ)
    words.json      เวลาระดับตัวอักษร/คำ ที่จับคู่กับเสียงจริง  <- ห้ามแก้
    edit.txt        ไฟล์กลางสำหรับย่อคำ                          <- แก้ตรงนี้
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from common import set_work_dir, fmt_ts, job_dir, load_api_key

# ชื่อย่อ -> (model id, คำอธิบายที่ผู้ใช้ควรรู้ก่อนเลือก)
GEMINI_MODELS = {
    "flash": (
        "gemini-flash-latest",
        "อยู่ใน free tier แน่นอน เร็ว แม่นพอ — ชื่อแบรนด์มักออกมาเป็นภาษาอังกฤษ",
    ),
    "pro": (
        "gemini-pro-latest",
        "แม่นกว่านิดหน่อย เขียนไทยล้วน — แต่โควต้า free tier น้อยกว่า อาจมีค่าใช้จ่าย",
    ),
    "flash-lite": (
        "gemini-flash-lite-latest",
        "เร็วสุดแต่ไม่แนะนำ — มักไม่เว้นวรรคและเขียนตัวเลขเป็นเลขอารบิก ทำให้แบ่งก้อนซับพลาด",
    ),
}


def resolve_model(choice: str) -> tuple[str, str]:
    """รับชื่อย่อหรือ model id เต็มก็ได้"""
    if choice in GEMINI_MODELS:
        return GEMINI_MODELS[choice]
    return choice, "(model id ที่ระบุเอง)"

PROMPT = (
    "ถอดเสียงภาษาไทยในไฟล์นี้แบบคำต่อคำ ทุกคำที่ได้ยินจริง รวมคำอุทาน เสียงเอ่อ อ้า "
    "คำพูดซ้ำ และประโยคที่พูดไม่จบ ห้ามสรุป ห้ามย่อ ห้ามแก้ไวยากรณ์ ห้ามข้ามช่วงใด "
    "ถ้าฟังไม่ออกให้ใส่ [ไม่ชัด] แทนการเดา\n"
    "ตัวเลขให้เขียนเป็นตัวหนังสือตามที่ได้ยิน เช่น แปดชั่วโมง ไม่ใช่ 8 ชั่วโมง "
    "(ข้อความต้องตรงกับเสียงคำต่อคำ เพราะจะถูกนำไปจับคู่กับเสียงจริง)\n"
    "รูปแบบผลลัพธ์: ข้อความล้วน ห้ามใส่ timestamp ห้ามใส่ป้ายผู้พูด ห้ามใส่หัวข้อ "
    "ห้ามใส่คำอธิบายใดๆ นอกเหนือจากข้อความที่ถอดได้\n"
    "ให้เว้นวรรคตรงจุดที่ผู้พูดหยุดหายใจหรือเว้นจังหวะจริง และขึ้นบรรทัดใหม่เมื่อจบประโยค "
    "(จุดเว้นวรรคเหล่านี้จะถูกใช้จับคู่กับเสียง ความแม่นของซับขึ้นกับตรงนี้)"
)


def parse_clock(s: str) -> float:
    """รับ 90 / 1:30 / 12:30.5 / 1:02:30 -> วินาที"""
    parts = str(s).strip().split(":")
    if len(parts) > 3:
        raise SystemExit(f"รูปแบบเวลาไม่ถูกต้อง: {s}")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def clock_slug(s: str) -> str:
    return str(s).replace(":", "").replace(".", "p")


def extract_audio(video: Path, out: Path, start: str | None, end: str | None) -> None:
    if out.exists():
        print(f"[1/4] มี {out.name} อยู่แล้ว ข้าม")
        return

    span = ""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(parse_clock(start))]
    cmd += ["-i", str(video)]
    if end:
        dur = parse_clock(end) - (parse_clock(start) if start else 0.0)
        if dur <= 0:
            raise SystemExit("--to ต้องมากกว่า --from")
        cmd += ["-t", str(dur)]
        span = f" ช่วง {start or '0:00'} ถึง {end} ({dur:.1f} วินาที)"
    elif start:
        span = f" ตั้งแต่ {start} จนจบไฟล์"

    print(f"[1/4] ดึงเสียงจาก {video.name}{span}")
    subprocess.run(cmd + ["-vn", "-ac", "1", "-ar", "16000", str(out)], check=True)


def transcribe(audio: Path, out: Path, model_choice: str) -> str:
    if out.exists() and out.read_text().strip():
        print(f"[2/4] มี {out.name} อยู่แล้ว ข้าม (ลบไฟล์ถ้าอยากถอดใหม่)")
        return out.read_text()

    from google import genai
    from google.genai import types

    model_id, note = resolve_model(model_choice)
    print(f"[2/4] ส่งให้ Gemini ถอดข้อความ — {model_choice} ({model_id})")
    print(f"      {note}")
    client = genai.Client(
        api_key=load_api_key(),
        http_options=types.HttpOptions(timeout=10 * 60 * 1000),
    )
    uploaded = client.files.upload(file=str(audio))
    resp = client.models.generate_content(model=model_id, contents=[uploaded, PROMPT])
    text = (resp.text or "").strip()
    if not text:
        raise SystemExit("Gemini ตอบกลับว่าง — ลองรันใหม่")
    out.write_text(text + "\n")
    print(f"      ได้ {len(text)} ตัวอักษร")
    return text


def align(audio: Path, text: str, model_name: str, device: str, max_chars: int, max_gap: float):
    import os
    import certifi
    # Python บน macOS ไม่ได้ใช้ cert ของระบบ ทำให้โหลด Whisper model ครั้งแรกพัง
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import stable_whisper

    print(f"[3/4] โหลด Whisper '{model_name}' ({device}) แล้วจับคู่ข้อความกับเสียง")
    model = stable_whisper.load_model(model_name, device=device)
    result = model.align(str(audio), text, language="th")
    if result is None:
        raise SystemExit("align ไม่สำเร็จ — ลองใช้ --model large-v3 หรือตรวจ transcript.txt")

    # จัดกลุ่มใหม่ให้เป็นก้อนขนาดซับ: ตัดตรงที่เงียบ แล้วคุมความยาวบรรทัด
    words = [
        {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
        for seg in result.segments for w in seg.words
    ]
    return words, group_into_cues(words, max_chars, max_gap)


def group_into_cues(words: list[dict], max_chars: int, max_gap: float) -> list[dict]:
    """รวมคำเป็นก้อนซับ โดยตัดได้เฉพาะตรงรอยเว้นวรรคเท่านั้น

    Whisper ตัด token ภาษาไทยเป็นตัวอักษรเดี่ยว การตัดตามจำนวนตัวอักษรตรง ๆ
    จึงทำให้คำขาดกลาง เราเลยจับกลุ่มเป็น "วลี" ตามช่องว่างที่ผู้พูดเว้นจริงก่อน
    แล้วค่อยแพ็ควลีเข้าก้อนให้พอดีความยาวบรรทัด
    """
    phrases, cur = [], []
    for w in words:
        if cur and w["word"].startswith((" ", "\n")):
            phrases.append(cur)
            cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)

    def text_of(ws):
        return " ".join("".join(x["word"] for x in ws).split())

    cues, bucket = [], []
    for ph in phrases:
        gap = ph[0]["start"] - bucket[-1]["end"] if bucket else 0.0
        too_long = len(text_of(bucket)) + 1 + len(text_of(ph)) > max_chars
        if bucket and (too_long or gap > max_gap):
            cues.append(bucket)
            bucket = []
        bucket.extend(ph)
    if bucket:
        cues.append(bucket)

    return [
        {"start": c[0]["start"], "end": c[-1]["end"], "text": text_of(c)}
        for c in cues if text_of(c)
    ]


def write_outputs(words: list[dict], cues: list[dict], d: Path) -> None:
    (d / "words.json").write_text(
        json.dumps({"words": words}, ensure_ascii=False, indent=1)
    )

    lines = [
        "# ไฟล์กลางสำหรับย่อคำ — แก้เฉพาะบรรทัดข้อความ (บรรทัดที่ไม่ขึ้นต้นด้วย #)",
        "#",
        "# ลบคำที่ไม่อยากให้ขึ้นได้เลย เวลาจะถูกคำนวณใหม่จากคำที่เหลือโดยอัตโนมัติ",
        "# อยากตัดซับก้อนนี้ทิ้ง  -> ลบข้อความจนเหลือบรรทัดว่าง",
        "# อยากแยกเป็นสองซับ     -> ใส่ | คั่นกลาง",
        "# ห้ามเพิ่มคำที่ไม่มีในเสียง (จะจับเวลาไม่ได้ ต้องใช้เวลาเดิมของก้อนแทน)",
        "#",
        "# เสร็จแล้วรัน: python3 make_srt.py " + shlex.quote(d.name),
        "",
    ]
    for i, c in enumerate(cues, 1):
        lines.append(f"[{i}] {fmt_ts(c['start'], '.')} → {fmt_ts(c['end'], '.')}")
        lines.append(c["text"])
        lines.append("")
    (d / "edit.txt").write_text("\n".join(lines))

    print(f"[4/4] เขียน words.json ({len(words)} token) และ edit.txt ({len(cues)} ก้อน)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="ถอดเสียงด้วย Gemini แล้ว align กับเสียงจริง",
        epilog=(
            "โมเดลที่ใช้ถอดข้อความ (--gemini-model):\n"
            + "".join(f"  {k:<11} {v[1]}\n" for k, v in GEMINI_MODELS.items())
            + "  หรือใส่ model id เต็มเองก็ได้ เช่น gemini-2.5-flash\n\n"
            "โมเดลมีผลแค่ 'ข้อความ' — ส่วน 'เวลา' มาจาก Whisper บนเครื่องนี้เสมอ\n"
            "ถ้าเปลี่ยนโมเดลแล้วอยากถอดใหม่ ต้องลบ transcript.txt ของงานนั้นก่อน"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="ไฟล์วิดีโอหรือไฟล์เสียง")
    p.add_argument("--name", help="ชื่อโฟลเดอร์งานใน work/ (ค่าเริ่มต้น: ชื่อไฟล์ + ช่วงเวลา)")
    p.add_argument("--from", dest="start", metavar="เวลา",
                   help="ตัดเฉพาะช่วง เริ่มที่ (เช่น 12:30) — เวลาใน SRT จะนับ 0 ที่จุดนี้")
    p.add_argument("--to", dest="end", metavar="เวลา",
                   help="ตัดเฉพาะช่วง จบที่ (เช่น 13:05)")
    p.add_argument("--gemini-model", default="flash", metavar="ชื่อ",
                   help="โมเดลที่ใช้ถอดข้อความ: flash (เริ่มต้น) / pro / flash-lite — ดูความต่างท้าย --help")
    p.add_argument("--model", default="medium", help="ขนาด Whisper: small/medium/large-v3 (เริ่มต้น medium)")
    p.add_argument("--device", default="cpu", help="cpu / mps / cuda (เริ่มต้น cpu — เสถียรสุดบน Mac)")
    p.add_argument("--max-chars", type=int, default=42, help="ตัวอักษรสูงสุดต่อซับหนึ่งก้อน")
    p.add_argument("--max-gap", type=float, default=0.4, help="ช่วงเงียบ (วินาที) ที่ถือว่าให้ตัดก้อนใหม่")
    p.add_argument("--work", metavar="โฟลเดอร์", help="โฟลเดอร์เก็บงาน (เริ่มต้น: ./work)")
    p.add_argument("--no-srt", action="store_true",
                   help="หยุดที่ edit.txt ไม่ต้องสร้าง SRT ให้ (ใช้เมื่อจะย่อคำก่อน)")
    args = p.parse_args()
    set_work_dir(args.work)

    video = Path(args.video).expanduser()
    if not video.exists():
        raise SystemExit(f"ไม่พบไฟล์: {video}")

    name = args.name
    if not name:
        name = video.stem
        if args.start or args.end:
            name += f"_{clock_slug(args.start or '0')}-{clock_slug(args.end or 'end')}"
    d = job_dir(name)
    audio = d / "audio.flac"

    extract_audio(video, audio, args.start, args.end)
    text = transcribe(audio, d / "transcript.txt", args.gemini_model)
    words, cues = align(audio, text, args.model, args.device, args.max_chars, args.max_gap)
    write_outputs(words, cues, d)

    print()
    if args.no_srt:
        print(f"เสร็จ → {d}")
        print(f"  1. เปิด {d / 'edit.txt'} แล้วย่อคำตามต้องการ")
        print(f"  2. รัน: python3 make_srt.py {shlex.quote(name)}")
    else:
        import make_srt
        cues = make_srt.build(name, d)
        srt = d / f"{name}.srt"
        make_srt.write_srt(cues, srt)
        print(f"เสร็จ → {srt}  ({len(cues)} ซับ)")
        print("ลากเข้า CapCut ได้เลย")
        print()
        print("ถ้าอยากย่อคำให้สั้นลงกว่าที่พูด เลือกได้ 2 ทาง:")
        print("  • แก้ไม่กี่ก้อน — ตัดใน CapCut ได้เลย")
        print("    แต่ต้องลากขอบซับให้ตรงคำที่เหลือด้วย ไม่งั้นจะขึ้นเร็วไปตามเวลาเดิมของก้อน")
        print(f"  • ย่อทั้งคลิป — แก้ {d / 'edit.txt'} แล้วรัน")
        print(f"    python3 make_srt.py {shlex.quote(name)}   (เวลาคำนวณใหม่ให้อัตโนมัติ)")
    if args.start:
        print(f"\nหมายเหตุ: เวลาใน SRT นับ 0 ที่ {args.start} ของไฟล์ต้นฉบับ")
        print("  ใช้กับคลิปที่ตัดหัวท้ายมาจากช่วงนี้เท่านั้น — ถ้ามี jump cut ข้างใน")
        print("  ต้อง export คลิปที่ตัดเสร็จแล้วมา align ใหม่")


if __name__ == "__main__":
    main()
