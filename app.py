# =========================================
# 🐄 MAWESHI DOST - HF SPACES VERSION
# =========================================

import os
import gradio as gr
from google import genai
from openai import OpenAI
from PIL import Image
import whisper
from gtts import gTTS
import tempfile
import traceback
import time

# =========================
# 🔑 API KEYS (from HF Secrets)
# =========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# =========================
# 🤖 INIT CLIENTS
# =========================
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

whisper_model = whisper.load_model("base")

# =========================
# 🎤 SPEECH → TEXT
# =========================
def speech_to_text(audio):
    if audio is None:
        return ""
    try:
        return whisper_model.transcribe(audio, language="ur")["text"]
    except:
        return ""

# =========================
# 🔊 TEXT → SPEECH
# =========================
def text_to_speech(text, language):
    try:
        if not text.strip():
            return None
        lang = "ur" if language == "Urdu" else "en"
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except:
        return None

# =========================
# 📷 GEMINI IMAGE ANALYSIS
# =========================
def analyze_image(image):
    if image is None:
        return ""

    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        prompt = """
        Analyze this livestock image carefully.

        Return:
        - Animal type (cow, buffalo, goat, sheep, chicken ONLY)
        - Visible symptoms
        - Health condition clues

        IMPORTANT:
        - Never say dog or cat
        - Be accurate and short
        """

        res = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[prompt, image]
        )

        return res.text

    except Exception as e:
        return ""

# =========================
# 🧠 GROQ REASONING
# =========================
def groq_generate(prompt):
    for _ in range(3):
        try:
            res = groq_client.responses.create(
                model="openai/gpt-oss-20b",
                input=prompt
            )
            return res.output_text
        except Exception as e:
            if "429" in str(e):
                time.sleep(5)
                continue
            return f"API ERROR: {str(e)}"
    return "System busy"

# =========================
# 🧠 PROMPT
# =========================
PROMPT = """
You are a livestock assistant helping farmers.

IMPORTANT:
- Image data may be imperfect
- Choose animal ONLY from: cow, buffalo, goat, sheep, chicken
- Keep field names EXACT

Language: {language}

Format:

Animal Type: ...
Disease: ...
Medicine: ...
Treatment: ...
Recovery Time: ...
When to See Vet: ...

Input:
{input}
"""

# =========================
# 🧠 PARSER
# =========================
def parse_response(text):
    keys = ["Animal Type","Disease","Medicine","Treatment","Recovery Time","When to See Vet"]
    data = {k:"" for k in keys}

    for line in text.split("\n"):
        for k in keys:
            if k.lower() in line.lower():
                parts = line.split(":",1)
                if len(parts)>1:
                    data[k] = parts[1].strip()
    return data

# =========================
# 🔍 MAIN FUNCTION
# =========================
def analyze(text_input, audio_input, image, language):
    try:
        voice = speech_to_text(audio_input)

        image_info = analyze_image(image)

        final_input = f"""
        Symptoms: {text_input}
        Voice: {voice}
        Image: {image_info}
        """

        if not final_input.strip():
            return "❌ No input provided", None

        lang = language

        prompt = PROMPT.format(
            input=final_input[:400],
            language=lang
        )

        ai_text = groq_generate(prompt)
        data = parse_response(ai_text)

        if all(v=="" for v in data.values()):
            return ai_text, text_to_speech(ai_text, language)

        formatted = f"""
### 🐄 {data['Animal Type']}
### 🦠 {data['Disease']}
### 💊 {data['Medicine']}
### 💉 {data['Treatment']}
### 📅 {data['Recovery Time']}
### ⚠️ {data['When to See Vet']}
"""

        audio = text_to_speech(ai_text, language)

        return formatted, audio

    except Exception as e:
        return f"❌ ERROR:\n{str(e)}\n{traceback.format_exc()}", None

# =========================
# 🎨 UI
# =========================
demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(label="Symptoms"),
        gr.Audio(sources=["microphone","upload"], type="filepath"),
        gr.Image(type="pil"),
        gr.Radio(["English","Urdu","Roman Urdu"], value="Urdu")
    ],
    outputs=[gr.Markdown(), gr.Audio()],
    title="🐄 Maweshi Dost AI",
)

demo.launch()
