# Iklil: Fanar Podcast Generator 🎙️👑

**Iklil** is an AI-powered Arabic podcast generation framework that leverages the **Fanar LLM** and **AURA speech technology** to create culturally rich, high-quality spoken content. The system is designed to automatically generate insightful, expressive, and natural-sounding Arabic podcasts—from knowledge-grounded scripts to expressive speech synthesis.

---

## 🌟 Project Vision

The goal of Iklil is to democratize Arabic podcast production by automating the pipeline from script generation to expressive audio output. The application focuses on culturally relevant topics, everyday knowledge, and conversational storytelling, ensuring naturalness, voice diversity, and linguistic depth.

---

## ⚠️ Challenges Addressed

### 1. **Content Depth and Insight Generation**
- Ensuring that generated podcasts are informative and grounded in reliable sources.
- Balancing entertainment with educational or cultural value.

### 2. **Natural Dialogue Generation**
- Creating flowing, engaging multi-turn conversations or monologues.
- Avoiding robotic or templated outputs through spontaneous script design.

### 3. **Appropriate Voice Representation**
- Matching vocal tone, gender, and accent with the content’s style and intended audience.
- Preserving dialectal diversity and identity in Arabic voices.

### 4. **Speech Quality and Expressiveness**
- Avoiding monotone delivery; achieving smooth prosody and emotional realism.
- Ensuring audio quality meets broadcasting standards.

---

## 🔧 Proposed Development Steps

### 1. **Knowledge Sources** [TODO Later]
- Curate structured and unstructured Arabic content from encyclopedic, journalistic, and conversational domains.
- Use Fanar LLM to summarize, extract, or expand culturally relevant material.

### 2. **Script Generation Module** [Current Task]
- Generate coherent, topical scripts based on input themes or daily prompts.
- Dialogue options: narrative, dialogue, interviews, news bulletin. Style options: Formal, Informal, ...
- Incontext Learning (few-shot) for spontaneity and variation (e.g., rhetorical devices, hesitations, digressions).

### 3. **Spontaneous Script Modeling** [Current Task]
- Incorporate disfluencies, fillers, or interjections to mimic real human delivery.
- Use few-shot prompting based tuning to model natural delivery.

### 4. **Audio and Speech Generation**
- Convert scripts to speech using the **AURA TTS engine**, with control over style, [tempo, emotion] and speaker identity.
- Allow speaker selection or customization.

### 5. **Audio Modeling Module**
- Apply post-processing: smoothing, dynamic range compression, and sound effects (e.g., intro music).
- Ensure alignment between emotion in speech and content of the script.

### 6. **Spontaneous Audio Refinement**
- Introduce variability across recordings of similar scripts.
- Experiment with expressive audio models trained on broadcast, conversational, and dialectal Arabic corpora.

---

## 🚀 Future Plans

- Add multilingual support (MSA and dialects).
- Deploy as a web or mobile interface with real-time podcast generation.
- Enable human-in-the-loop editing for semi-automated podcasts.

---

## 🤝 Contributing

We welcome contributors with interests in:
..
To get involved, please reach out or open an issue!

---

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

---

## 👤 Acknowledgments

Built by the <internship team> .. to add
 
**Fanar LLM** and **AURA Speech** at [QCRI](https://www.qcri.org/), combining state-of-the-art Arabic language models and expressive speech synthesis technologies.


