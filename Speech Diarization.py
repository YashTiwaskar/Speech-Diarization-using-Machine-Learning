# -*- coding: utf-8 -*-
"""Speech_Model.ipynb"""

import os
import torchaudio

def convert_to_16k_mono(root_dir):
    print("Converting audio files to 16kHz mono...")
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".wav"):
                filepath = os.path.join(subdir, filename)
                print(f"Found file: {filepath}")
                try:
                    waveform, sample_rate = torchaudio.load(filepath)
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    # Resample if needed
                    if sample_rate != 16000:
                        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
                    # Save over original file in 16kHz mono
                    torchaudio.save(filepath, waveform, 16000)
                    print(f"Converted: {filepath}")
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")

# Run conversion on sample_data folder
convert_to_16k_mono("sample_data")

# Step 1: Install dependencies (uncomment if running in Colab)
# !pip install -q speechbrain torchaudio xgboost scikit-learn joblib

# Step 2: Import libraries
import numpy as np
import torch
from speechbrain.pretrained import SpeakerRecognition
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import joblib

# Load speaker recognition model from SpeechBrain
classifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir"
)

# Dataset path
dataset_path = "/content/sample_data"

# Step 3: Extract embeddings and labels
X, y = [], []
segment_duration = 3  # seconds

print("\nExtracting embeddings from audio files...")

for speaker_folder in os.listdir(dataset_path):
    speaker_path = os.path.join(dataset_path, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue

    for file in os.listdir(speaker_path):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(speaker_path, file)

        try:
            # Load and preprocess audio
            signal, fs = torchaudio.load(file_path)

            # Convert stereo to mono
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

            # Resample to 16kHz
            if fs != 16000:
                signal = torchaudio.transforms.Resample(fs, 16000)(signal)
                fs = 16000

            duration = signal.shape[1] / fs
            num_segments = int(duration // segment_duration)

            if num_segments == 0:
                print(f"Skipping too short: {file_path}")
                continue

            for i in range(num_segments):
                start_sample = int(i * segment_duration * fs)
                end_sample = int((i + 1) * segment_duration * fs)
                segment = signal[:, start_sample:end_sample]

                if segment.shape[1] == 0:
                    continue

                embedding = classifier.encode_batch(segment)
                X.append(embedding.squeeze().numpy())
                y.append(speaker_folder)

            print(f"Processed {num_segments} segments: {file_path}")

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

# Encode string labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split for balanced evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.5, stratify=y_encoded, random_state=42
)

# Show class distribution (decoded)
print("\nTrain distribution:", Counter(le.inverse_transform(y_train)))
print("Test distribution:", Counter(le.inverse_transform(y_test)))

# Train XGBoost classifier
print("\nTraining XGBoost classifier...")
clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
clf.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

# Save the classifier and label encoder
joblib.dump(clf, "/content/xgb_classifier.joblib")
joblib.dump(le, "/content/label_encoder.joblib")
print("XGBoost classifier saved as xgb_classifier.joblib")
print("Label encoder saved as label_encoder.joblib")

# === ADDITIONAL DEPENDENCIES (install if needed) ===
# !pip install -q speechbrain torchaudio xgboost scikit-learn joblib
# !pip install git+https://github.com/openai/whisper.git
# !pip install git+https://github.com/m-bain/whisperx.git
# !pip install git+https://github.com/snakers4/silero-vad.git
# !pip install openpyxl gender-guesser ffmpeg-python

# === IMPORTS FOR INFERENCE ===
import sys
import ffmpeg
import whisperx
import whisper
import pandas as pd
from pathlib import Path
import gender_guesser.detector as gender
from google.colab import files
from whisperx.diarize import DiarizationPipeline
from speechbrain.pretrained import EncoderClassifier

# Load Silero VAD
vad_model = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True, trust_repo=True)[0]
utils_path = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master" / "src" / "silero_vad"
sys.path.append(str(utils_path))
from silero_vad import get_speech_timestamps, read_audio

# Upload audio file
uploaded = files.upload()
if not uploaded:
    raise Exception("No file uploaded.")
audio_file = list(uploaded.keys())[0]

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# Preprocessing audio with FFmpeg
print("Preprocessing audio with FFmpeg...")
cleaned_audio = "cleaned_audio.wav"
ffmpeg.input(audio_file).output(
    cleaned_audio, ac=1, ar=16000, af='loudnorm', y=None
).run(quiet=True)

# Voice Activity Detection (VAD)
print("Running Silero VAD...")
wav = read_audio(cleaned_audio, sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
print(f"Detected {len(speech_timestamps)} speech segments.")

# Language Detection
lang_model = whisper.load_model("base")
audio_wav = whisper.load_audio(cleaned_audio)
audio_wav = whisper.pad_or_trim(audio_wav)
mel = whisper.log_mel_spectrogram(audio_wav).to(lang_model.device)
_, lang_probs = lang_model.detect_language(mel)
detected_lang = max(lang_probs, key=lang_probs.get)
if detected_lang == "ur":
    detected_lang = "hi"
print(f"Detected Language: {detected_lang} ({lang_probs[detected_lang]:.2f})")

# Transcription
model = whisperx.load_model("medium", device=device, compute_type=compute_type)
transcription = model.transcribe(
    cleaned_audio, batch_size=8, language=detected_lang
)

# Alignment
model_a, metadata = whisperx.load_align_model(
    language_code=detected_lang, device=device
)
aligned_result = whisperx.align(
    transcription["segments"], model_a, metadata, cleaned_audio, device
)

# Diarization
diarize_model = DiarizationPipeline(use_auth_token="YOUR_HF_TOKEN")
diarize_segments = diarize_model(cleaned_audio)

# Speaker Embeddings and XGBoost Classification
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec"
)
clf = joblib.load("/content/xgb_classifier.joblib")
label_encoder = joblib.load("/content/label_encoder.joblib")

def extract_embedding(audio_path, start, end):
    signal, sr = torchaudio.load(audio_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
        signal = torchaudio.transforms.Resample(sr, 16000)(signal)
    segment = signal[:, int(start * 16000):int(end * 16000)]
    if segment.shape[1] == 0:
        return np.zeros((192,))
    embedding = classifier.encode_batch(segment)
    return embedding.squeeze().detach().cpu().numpy()

# Predict speaker name using label_encoder.inverse_transform
for i, row in diarize_segments.iterrows():
    emb = extract_embedding(cleaned_audio, row['start'], row['end'])
    pred_label = clf.predict(emb.reshape(1, -1))[0]
    pred_name = label_encoder.inverse_transform([pred_label])[0]
    diarize_segments.at[i, 'speaker'] = pred_name

# Merge transcript with speaker segments
def merge_text_with_speakers(segments, speaker_segments_df):
    output = []
    speaker_segments_list = speaker_segments_df.to_dict('records')
    i = 0
    for seg in segments:
        while i < len(speaker_segments_list) and speaker_segments_list[i]['end'] <= seg['start']:
            i += 1
        speaker = "unknown"
        if i < len(speaker_segments_list):
            s = speaker_segments_list[i]
            if s['start'] <= seg['start'] <= s['end']:
                speaker = s['speaker']
            elif (seg['start'] < s['start'] and i > 0 and
                  speaker_segments_list[i - 1]['start'] <= seg['start'] <= speaker_segments_list[i - 1]['end']):
                speaker = speaker_segments_list[i - 1]['speaker']
        seg['speaker'] = speaker
        output.append(seg)
    return output

final_output = merge_text_with_speakers(aligned_result['segments'], diarize_segments)

# Format and export to Excel
gen = gender.Detector()
rows = []
for seg in final_output:
    pred_name = seg['speaker']
    gender_guess = gen.get_gender(pred_name)
    gender_final = "male" if "male" in gender_guess else "female" if "female" in gender_guess else "unknown"
    rows.append({
        "file_name": audio_file,
        "start_time": round(seg['start'], 2),
        "end_time": round(seg['end'], 2),
        "speaker": pred_name,
        "speaker_name": pred_name,
        "gender": gender_final,
        "text": seg['text']
    })
df = pd.DataFrame(rows)
df.to_excel("whisperx_diarization_output.xlsx", index=False)
print("Output saved to 'whisperx_diarization_output.xlsx'")
