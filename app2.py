import streamlit as st
import os
import shutil
import zipfile
import torchaudio
import torch
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from speechbrain.pretrained import EncoderClassifier
from whisperx.diarize import DiarizationPipeline
import whisperx
import ffmpeg
import gender_guesser.detector as gender

# Set page config
st.set_page_config(page_title="Speaker Diarization & Recognition App", layout="wide")

st.title("🎙 Speaker Diarization & Recognition App")

# Setup device and compute_type properly
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"  # Use float32 on CPU

# Hardcoded Hugging Face token (Replace with your own token)
HF_TOKEN = "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"

# Cache speaker embedding model loading to improve performance
@st.cache_resource(show_spinner=False)
def load_speaker_embedding_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_speaker_embedding_model",
    )
classifier = load_speaker_embedding_model()

gen = gender.Detector()

# --- Section 1: Upload Training Data ---
st.header("Step 1: Upload Training Data ZIP (One folder per speaker)")
train_zip = st.file_uploader("Upload zipped training folder", type="zip", key="trainzip")

# Initialize session state placeholders for XGBoost classifier and label encoder
if "xgb_clf" not in st.session_state:
    st.session_state["xgb_clf"] = None
if "label_encoder" not in st.session_state:
    st.session_state["label_encoder"] = None

def extract_zip(zip_file, dest_folder):
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(dest_folder)

def extract_embeddings(folder_path, segment_duration=3):
    X, y = [], []
    for spk in os.listdir(folder_path):
        spk_path = os.path.join(folder_path, spk)
        if not os.path.isdir(spk_path):
            continue
        for wav_file in os.listdir(spk_path):
            if not wav_file.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(spk_path, wav_file)
            try:
                signal, sr = torchaudio.load(wav_path)
                if signal.shape[0] > 1:
                    signal = signal.mean(dim=0, keepdim=True)
                if sr != 16000:
                    signal = torchaudio.transforms.Resample(sr, 16000)(signal)
                    sr = 16000
                duration_seconds = signal.shape[1] / sr
                num_segments = int(duration_seconds // segment_duration)
                for i in range(num_segments):
                    start_sample = int(i * segment_duration * sr)
                    end_sample = int((i + 1) * segment_duration * sr)
                    segment = signal[:, start_sample:end_sample]
                    if segment.shape[1] == 0:
                        continue
                    emb = classifier.encode_batch(segment).squeeze().detach().cpu().numpy()
                    X.append(emb)
                    y.append(spk)
            except Exception as e:
                st.warning(f"Skipping {wav_path}: {e}")
    return X, y

if train_zip is not None:
    train_dir = "train_data"
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)

    with open("train.zip", "wb") as f:
        f.write(train_zip.read())

    extract_zip("train.zip", train_dir)
    st.success("Training ZIP extracted.")

    with st.spinner("Extracting embeddings and training classifier..."):
        X, y = extract_embeddings(train_dir)
        if len(set(y)) < 2:
            st.error("At least 2 distinct speakers required for training.")
        elif len(X) == 0:
            st.error("No valid audio segments found for training.")
        else:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            clf.fit(X, y_encoded)

            st.session_state["xgb_clf"] = clf
            st.session_state["label_encoder"] = le

            joblib.dump(clf, "xgb_classifier.joblib")
            joblib.dump(le, "label_encoder.joblib")

            st.success(f"Trained classifier on {len(set(y))} speakers with {len(X)} segments.")

# --- Section 2: Upload Audio to Diarize & Transcribe ---
st.header("Step 2: Upload Audio to Diarize & Transcribe")
test_audio = st.file_uploader("Upload audio file for diarization", type=["wav"], key="testaudio")

if st.button("Run Diarization & Speaker Identification"):
    if test_audio is None:
        st.error("Please upload an audio file to diarize.")
    elif st.session_state["xgb_clf"] is None or st.session_state["label_encoder"] is None:
        st.error("Please upload training data and train the classifier first.")
    else:
        with open("test_audio.wav", "wb") as f:
            f.write(test_audio.read())
        st.success("Test audio saved.")

        cleaned_audio = "cleaned_test_audio.wav"
        try:
            ffmpeg.input("test_audio.wav").output(
                cleaned_audio, ac=1, ar=16000, af="loudnorm", y=None
            ).run(quiet=True)
            st.success("Audio preprocessed to 16kHz mono.")
        except Exception as e:
            st.error(f"Error preprocessing audio: {e}")
            st.stop()

        st.info("Running diarization...")
        try:
            diarize_model = DiarizationPipeline(
                use_auth_token=HF_TOKEN,
                device=device,
            )
            diarize_segments = diarize_model(cleaned_audio)
            if "segment" in diarize_segments.columns:
                diarize_segments["segment"] = diarize_segments["segment"].astype(str)
            st.write(diarize_segments)
        except Exception as e:
            st.error(f"Diarization error: {e}")
            st.stop()

        st.info("Transcribing audio...")
        try:
            # Use whisperx's transcribe (without manual log_mel_spectrogram)
            model_base = whisperx.load_model("base", device=device, compute_type=compute_type)
            transcription_base = model_base.transcribe(cleaned_audio)
            detected_lang = transcription_base.get("language", None)
            if detected_lang is None:
                detected_lang = "en"
            st.write(f"Detected language: {detected_lang}")

            model = whisperx.load_model("medium", device=device, compute_type=compute_type)
            transcription = model.transcribe(cleaned_audio, batch_size=2, language=detected_lang)
            st.write("Transcription segments:", transcription["segments"])

            model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
            aligned_result = whisperx.align(transcription["segments"], model_a, metadata, cleaned_audio, device)
        except Exception as e:
            st.error(f"Transcription error: {e}")
            st.stop()

        st.info("Recognizing speakers for each segment...")
        clf = st.session_state["xgb_clf"]
        le = st.session_state["label_encoder"]

        signal, sr = torchaudio.load(cleaned_audio)
        results = []

        for i, row in diarize_segments.iterrows():
            start_sample = int(row["start"] * sr)
            end_sample = int(row["end"] * sr)
            segment = signal[:, start_sample:end_sample]
            if segment.shape[1] > 0:
                emb = classifier.encode_batch(segment).squeeze().detach().cpu().numpy()
                pred_label = clf.predict(emb.reshape(1, -1))[0]
                pred_name = le.inverse_transform([pred_label])[0]
                gender_guess = gen.get_gender(pred_name)
                gender_final = (
                    "male"
                    if "male" in gender_guess
                    else "female"
                    if "female" in gender_guess
                    else "unknown"
                )
            else:
                pred_name = "unknown"
                gender_final = "unknown"

            results.append(
                {
                    "start_time": round(row["start"], 3),
                    "end_time": round(row["end"], 3),
                    "speaker": pred_name,
                    "gender": gender_final,
                }
            )

        df_res = pd.DataFrame(results)
        st.write("Speaker labeled segments:", df_res)

        output = BytesIO()
        df_res.to_excel(output, index=False)
        st.download_button(
            label="Download diarization results as XLSX",
            data=output.getvalue(),
            file_name="diarization_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

