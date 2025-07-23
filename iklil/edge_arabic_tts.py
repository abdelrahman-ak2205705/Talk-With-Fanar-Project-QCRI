import re, os, asyncio, subprocess, edge_tts
from pathlib import Path
import asyncio

SCRIPT_PATH = "fanar-c_script.txt"   # تأكد من الاسم الصحيح

# 1️⃣ إعداد الأصوات (يمكنك تغييرها كما تريد)
VOICE = {
    "المقدم": "ar-KW-FahedNeural",
    "الضيف" : "ar-BH-LailaNeural",
}

# 2️⃣ دالة قراءة المقاطع
def load_chunks(path=SCRIPT_PATH):
    pattern = re.compile(r'^(\s*(?:المقدم|مقدّم|مقدم|الضيف|ضيف))\s*:\s*(.+)$')
    chunks = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                continue  # تجاهل أي أسطر لا تطابق النمط
            speaker_raw, text = m.groups()

            # توحيد مفتاح المتحدّث ليتفق مع dict VOICE
            speaker = "المقدم" if "قدم" in speaker_raw else "الضيف"

            # استبدال <pause:Ns> بـ SSML break
            text = re.sub(r'<pause:\s*(\d+)s>', r'<break time="\1s"/>', text)
            text = re.sub(r'<[^>]+>', '', text)  # إزالة أي وسم عشوائي آخر

            ssml = f"{text}"
            chunks.append((speaker, ssml))
    return chunks

# 3️⃣ التوليد وحفظ المقاطع
async def synthesize():
    chunks = load_chunks()
    for i, (speaker, ssml) in enumerate(chunks, 1):
        voice = VOICE[speaker]
        communicate = edge_tts.Communicate(ssml, voice)
        # اسم الملف بصيغة 001.mp3، 002.mp3 ...
        out_name = f"{i:03d}.mp3"
        await communicate.save(out_name)
        await asyncio.sleep(0.3)  # مهلة صغيرة لتجنّب معدّل طلب عالٍ

# 4️⃣ دمج جميع المقاطع في ملف واحد
def concat_mp3(output="fanar-c-podcast.mp3"):
    mp3_files = sorted(Path(".").glob("???.mp3"))
    with open("filelist.txt", "w", encoding="utf-8") as f:
        for mp3 in mp3_files:
            f.write(f"file '{mp3.as_posix()}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "filelist.txt",
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        "-ar", "44100",
        output
    ], check=True)

    os.remove("filelist.txt")
    print(f"✅ ملف البودكاست النهائي: {output}")

# 5️⃣ تشغيل كل شيء
async def main():
    await synthesize()
    concat_mp3()

if __name__ == "__main__":
    asyncio.run(main())
