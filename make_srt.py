#!/usr/bin/env python3
"""ขั้นที่ 2: edit.txt (ข้อความที่ย่อแล้ว) + words.json (เวลาจริง) -> .srt

    python3 make_srt.py clip15

หัวใจอยู่ที่ locate_text(): ข้อความที่เหลือหลังย่อจะถูกจับคู่กลับไปหาเวลาของ
ตัวอักษรจริงในเสียง ซับจึงเริ่มตรงคำแรกที่เหลือ และจบตรงคำสุดท้ายที่เหลือ
ไม่ใช่กินเวลาของประโยคเต็มเหมือนตอนแก้ timestamp ด้วยมือ
"""
import argparse
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

from common import set_work_dir, build_char_timeline, fmt_ts, job_dir, load_words, normalize

HEADER_RE = re.compile(r"^\[(\d+)\]\s+([\d:.]+)\s*→\s*([\d:.]+)\s*$")


def parse_ts(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))


def parse_edit(path: Path) -> list[dict]:
    """อ่าน edit.txt -> [{idx, start, end, text}] (ก้อนที่ข้อความว่างจะถูกตัดทิ้ง)"""
    cues, current = [], None
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = HEADER_RE.match(line.strip())
        if m:
            if current:
                cues.append(current)
            current = {
                "idx": int(m.group(1)),
                "start": parse_ts(m.group(2)),
                "end": parse_ts(m.group(3)),
                "text": "",
            }
            continue
        if current is not None and line.strip():
            current["text"] = (current["text"] + " " + line.strip()).strip()
    if current:
        cues.append(current)
    return [c for c in cues if c["text"]]


def build_index(words):
    """char timeline ที่ตัดช่องว่างออกแล้ว — ใช้เทียบข้อความ"""
    text, starts, ends = build_char_timeline(words)
    chars, cs, ce = [], [], []
    for ch, s, e in zip(text, starts, ends):
        if ch.strip():
            chars.append(ch)
            cs.append(s)
            ce.append(e)
    return "".join(chars), cs, ce


def window_for(cs, ce, start: float, end: float, pad: float = 0.6):
    """ช่วง index ของตัวอักษรที่อยู่ในกรอบเวลาของก้อนเดิม (เผื่อขอบ)"""
    lo = next((i for i, e in enumerate(ce) if e >= start - pad), 0)
    hi = next((i for i in range(len(cs) - 1, -1, -1) if cs[i] <= end + pad), len(cs) - 1)
    return lo, min(hi + 1, len(cs))


def locate_text(edited: str, hay: str, cs, ce, lo: int, hi: int):
    """หาเวลาเริ่ม-จบของข้อความที่ย่อแล้ว ภายในหน้าต่าง [lo, hi)

    คืน (start, end, ratio) — ratio คือสัดส่วนตัวอักษรที่จับคู่ได้จริง
    ใช้ SequenceMatcher เพราะรองรับการลบคำจากกลางประโยคและการสลับเล็กน้อย
    """
    needle = normalize(edited)
    if not needle:
        return None
    window = hay[lo:hi]
    sm = SequenceMatcher(None, window, needle, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    first, last = blocks[0], blocks[-1]
    start = cs[lo + first.a]
    end = ce[lo + last.a + last.size - 1]
    return start, end, matched / len(needle)


def build_cues(edit_cues, hay, cs, ce, pad_window: float):
    out = []
    for c in edit_cues:
        lo, hi = window_for(cs, ce, c["start"], c["end"], pad_window)
        cursor = lo
        parts = [p.strip() for p in c["text"].split("|") if p.strip()]
        for part in parts:
            found = locate_text(part, hay, cs, ce, cursor, hi)
            if found and found[2] >= 0.5:
                start, end, ratio = found
                weak = ratio < 0.8
            else:
                start, end, weak = c["start"], c["end"], True
            out.append({"text": " ".join(part.split()), "start": start, "end": end,
                        "weak": weak, "src": c["idx"]})
            # เลื่อน cursor ไปหลังส่วนที่เพิ่งจับได้ กันจับซ้ำที่เดิมเมื่อมีคำซ้ำ
            if found:
                cursor = max(cursor, next((i for i, s in enumerate(cs) if s >= end), cursor))
    return out


def audio_duration(d: Path, words) -> float:
    """ความยาวคลิป — ใช้เป็นเพดาน ไม่ให้ซับก้อนท้ายล้นออกไปนอกคลิป"""
    audio = d / "audio.flac"
    if audio.exists():
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(audio)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            return float(out)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    return max(w["end"] for w in words)


def polish(cues, min_dur: float, lead: float, tail: float, gap: float, limit: float):
    """เผื่อหัวท้าย บังคับความยาวขั้นต่ำ แล้วกันซับซ้อนทับกัน"""
    for c in cues:
        c["start"] = max(0.0, c["start"] - lead)
        c["end"] = min(c["end"] + tail, limit)
        if c["end"] - c["start"] < min_dur:
            c["end"] = min(c["start"] + min_dur, limit)
    cues.sort(key=lambda c: c["start"])
    for a, b in zip(cues, cues[1:]):
        if a["end"] > b["start"] - gap:
            a["end"] = max(a["start"] + 0.2, b["start"] - gap)
    return cues


def write_srt(cues, path: Path) -> None:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(
            f"{i}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{c['text']}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="สร้าง SRT จากไฟล์ที่ย่อคำแล้ว")
    p.add_argument("name", help="ชื่องานใน work/")
    p.add_argument("-o", "--out", help="path ไฟล์ .srt (เริ่มต้น: work/<name>/<name>.srt)")
    p.add_argument("--min-dur", type=float, default=0.7, help="ความยาวขั้นต่ำต่อซับ (วินาที)")
    p.add_argument("--lead", type=float, default=0.06, help="เผื่อเวลาก่อนคำแรก (วินาที)")
    p.add_argument("--tail", type=float, default=0.18, help="เผื่อเวลาหลังคำสุดท้าย (วินาที)")
    p.add_argument("--gap", type=float, default=0.04, help="ช่องว่างขั้นต่ำระหว่างซับสองก้อน")
    p.add_argument("--window-pad", type=float, default=0.6, help="ขยายกรอบค้นหาของแต่ละก้อน (วินาที)")
    p.add_argument("--work", metavar="โฟลเดอร์", help="โฟลเดอร์เก็บงาน (เริ่มต้น: ./work)")
    args = p.parse_args()
    set_work_dir(args.work)

    d = job_dir(args.name)
    edit_path = d / "edit.txt"
    if not edit_path.exists():
        raise SystemExit(f"ไม่พบ {edit_path} — รัน align.py ก่อน")

    words = load_words(args.name)
    hay, cs, ce = build_index(words)
    edit_cues = parse_edit(edit_path)
    if not edit_cues:
        raise SystemExit("ไม่มีข้อความเหลือใน edit.txt เลย")

    cues = build_cues(edit_cues, hay, cs, ce, args.window_pad)
    cues = polish(cues, args.min_dur, args.lead, args.tail, args.gap,
                  audio_duration(d, words))

    out = Path(args.out) if args.out else d / f"{args.name}.srt"
    write_srt(cues, out)

    weak = [c for c in cues if c["weak"]]
    print(f"เขียน {out}  ({len(cues)} ซับ)")
    if weak:
        print(f"\nเตือน: {len(weak)} ก้อนจับคู่กับเสียงไม่ได้ชัด — ใช้เวลาของก้อนเดิมแทน")
        for c in weak[:10]:
            print(f"  [{c['src']}] {fmt_ts(c['start'], '.')}  {c['text'][:40]}")
        print("  (มักเกิดจากพิมพ์คำใหม่ที่ไม่มีในเสียง — แก้ให้ใช้คำเดิมแล้วรันซ้ำ)")


if __name__ == "__main__":
    main()
