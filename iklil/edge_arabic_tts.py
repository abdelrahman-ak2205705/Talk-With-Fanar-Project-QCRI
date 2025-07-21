import re, os, asyncio, subprocess, edge_tts

VOICE = {"المقدم": "ar-BH-LailaNeural",      
         "الضيف":  "ar-KW-FahedNeural"}     
         

def load_chunks(path="intro.txt"):
    chunks = []
    for line in open(path, "r", encoding="utf-8"):
        m = re.match(r'^(المقدم|الضيف):\s*(.*)$', line.strip())
        if not m:           
            continue
        speaker, text = m.groups()
        text = re.sub(r'<pause:\s*(\d+)s>', r'<break time="\1s"/>', text)
        text = re.sub(r'<[^>]+>', '', text)
        ssml = f"{text}"
        chunks.append((speaker, ssml))
    return chunks

async def synthesize():
    for i, (spk, ssml) in enumerate(load_chunks(), 1):
        voice = VOICE[spk]
        communicate = edge_tts.Communicate(ssml, voice)
        await communicate.save(f"{i:03d}.mp3")

asyncio.run(synthesize())

clips = "|".join(sorted(f for f in os.listdir() if f.endswith(".mp3")))
subprocess.run(["ffmpeg", "-y", "-i", f"concat:{clips}", "-c", "copy", "podcast_intro.mp3"])
print("🎙  Final file → podcast_intro.mp3")