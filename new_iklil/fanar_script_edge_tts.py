# -*- coding: utf-8 -*-
"""
make_podcast.py
===============

• يقرأ نصّ الحلقة من ملف script.txt (UTF‑8) في المجلد نفسه.
• يدمج صوتَين مختلفَين (المقدم ↔ الضيوف) في ملف podcast_ai_education.mp3
• المتطلبات:  pip install edge-tts
"""

import asyncio
import sys
import re
import html
from pathlib import Path

import edge_tts

# ----------------------------------------------------------------------
# 0) تأكّد أن الطرفيّة تطبع UTF‑8 على Windows لتفادي UnicodeEncodeError
# ----------------------------------------------------------------------
sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------
# 1) حمّل النصّ من script.txt
# ----------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).with_name("script2.txt")
RAW_SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# 2) خريطة المتحدث ↦ الصوت (أصوات Microsoft Arabic Neural)
# ----------------------------------------------------------------------
VOICE_MAP = {
    "المقدم":  "ar-SA-HamedNeural",    # صوت المقدم (ذكر)
    "default": "ar-SA-ZariyahNeural",  # صوت الضيوف (أنثى)
}

# ----------------------------------------------------------------------
# 3) حوّل كل سطر إلى مقطع (voice, text) مع تعقيم العلامات
# ----------------------------------------------------------------------
def build_segments(raw_text: str):
    segs = []
    for line in raw_text.splitlines():
        line = line.strip()

        # تجاهل العناوين والفواصل والملاحظات بين قوسين
        if not line or line.startswith(("---", "===")) or re.fullmatch(r"\(.+\)", line):
            continue

        m = re.match(r"^([^:]+):\s*(.+)$", line)
        speaker, text = (m.group(1).strip(), m.group(2).strip()) if m else ("المقدم", line)

        voice = VOICE_MAP.get(speaker, VOICE_MAP["default"])
        segs.append((voice, html.escape(text, quote=False)))
    return segs

# ----------------------------------------------------------------------
# 4) كوّن SSML من المقاطع
# ----------------------------------------------------------------------
def build_ssml(segments):
    parts = ['<speak version="1.0" xml:lang="ar-SA">']
    parts += [f'<voice name="{v}">{t}</voice>' for v, t in segments]
    parts.append('</speak>')
    return "\n".join(parts)

# ----------------------------------------------------------------------
# 5) توليد الملف الصوتي باستخدام edge‑tts
# ----------------------------------------------------------------------
async def synthesize(ssml: str, outfile: str = "podcast_ai2.mp3"):
    communicate = edge_tts.Communicate(ssml, voice="ar-SA-HamedNeural")  # الصوت هنا مبدئي فقط
    await communicate.save(outfile)
    print(f"✓ تمّ إنشاء الملف: {outfile}")

# ----------------------------------------------------------------------
# 6) نقطة التشغيل
# ----------------------------------------------------------------------
if __name__ == "__main__":
    segments = build_segments(RAW_SCRIPT)
    ssml = build_ssml(segments)
    asyncio.run(synthesize(ssml))
