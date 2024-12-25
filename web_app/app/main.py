## Links for ref:
# ASR model -> https://huggingface.co/openai/whisper-large-v3
# emotion detection docs -> https://speechbrain.github.io/ AND
# emotion detection hf model -> https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP

import os
import torch
import tempfile
import subprocess
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from speechbrain.inference.interfaces import foreign_class
import torchaudio
import time

# Add FFmpeg to PATH
os.environ["PATH"] += os.pathsep + "/home/qsh5523/miniconda3/envs/tereo/bin"
# Load Whisper ASR model with GPU support
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
print("Whisper model loaded successfully.")

# Load SpeechBrain Emotion Classifier
emotion_classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
print("SpeechBrain Emotion Classifier loaded successfully.")

# Directory to save audio files
AUDIO_DIR = "web_app/app/static/audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = Flask(__name__)

@app.route('/')
def dashboard():
    """Render the main transcription page."""
    return render_template('dashboard.html')

@app.route('/testasr')
def asr():
    """Render the main transcription page."""
    return render_template('testASR.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Save the uploaded audio to a temporary file
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
            webm_path = temp_webm.name
            audio_file.save(webm_path)

        # Convert .webm to .mp3
        mp3_filename = f"{AUDIO_DIR}/audio_{int(time.time())}.mp3"  # Unique filename
        ffmpeg_command = ["ffmpeg", "-i", webm_path, "-q:a", "0", mp3_filename]
        subprocess.run(ffmpeg_command, check=True)

        print(f"Saved audio file as MP3: {mp3_filename}")

        # Transcribe audio using Whisper
        result = pipe(mp3_filename)
        transcription = result["text"]
        print(f"Transcription result: {transcription}")

        # Emotion detection using SpeechBrain
        _, _, _, emotion_label = emotion_classifier.classify_file(mp3_filename)
        print(f"Emotion detected: {emotion_label}")

        return jsonify({
            "transcription": transcription,
            "emotion": emotion_label,
            "audio_file": mp3_filename
        })

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        files = request.files.getlist('files')
        transcriptions = []

        for file in files:
            # Ensure the directory exists
            os.makedirs(AUDIO_DIR, exist_ok=True)

            # Save the uploaded file
            input_path = os.path.join(AUDIO_DIR, file.filename)
            if os.path.exists(input_path):
                print(f"File {file.filename} already exists. Replacing it.")
            else:
                print(f"Saving new file: {file.filename}")

            # Save or replace the file
            try:
                file.save(input_path)  # Save the file to the specified path
                print(f"File saved to {input_path}")
            except Exception as e:
                print(f"Error saving file {file.filename}: {e}")
                return jsonify({"error": f"Failed to save file {file.filename}"}), 500

            # Convert .m4a to .wav using FFmpeg
            if input_path.endswith(".m4a"):
                output_path = os.path.splitext(input_path)[0] + ".wav"
                ffmpeg_command = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path]
                subprocess.run(ffmpeg_command, check=True)
                print(f"Converted {input_path} to {output_path}")
            else:
                output_path = input_path

            # Transcribe the audio file (Māori transcription)
            transcription_result = pipe(output_path)
            transcription = transcription_result["text"]

            # Translate the audio file (Māori to English translation)
            translation_result = pipe(output_path, generate_kwargs={"task": "translate"})
            translation = translation_result["text"]

            # Emotion detection
            _, _, _, emotion_label = emotion_classifier.classify_file(output_path)

            # Append results
            transcriptions.append(
                f"Transcription: {transcription}, \n"
                f"Translation to English: {translation}, \n"
                f"Emotion Detected from Audio: {emotion_label}"
            )

        # Join all transcriptions if multiple files are uploaded
        response_text = "\n\n".join(transcriptions)
        return jsonify({"transcription": response_text})

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)