# Speech-Diarization-using-Machine-Learning
The system uses advanced machine learning techniques, including voice activity detection (VAD), feature extraction, speaker embedding, and clustering algorithms to accurately distinguish multiple speakers in an audio stream.

Speech Diarization
This repository provides a comprehensive pipeline for speech diarization, speaker recognition, and speech transcription using state-of-the-art models such as WhisperX, Silero VAD, and SpeechBrain. It processes audio files, performs voice activity detection, transcribes speech with speaker labels, predicts gender, and exports the diarized results to an Excel file.

Features
Audio Preprocessing: Converts input audio to 16kHz mono WAV automatically.
Voice Activity Detection (VAD): Uses Silero VAD to detect speech segments.
Speech Transcription: High-quality transcription using WhisperX with automatic language detection.
Speaker Diarization: Speaker segmentation and labeling via SpeechBrain embeddings and a trained XGBoost classifier.
Gender Prediction: Infers speaker gender (male, female, unknown) using the speaker name.
Output: Saves diarized transcripts with speaker and gender info into an Excel .xlsx file.
Custom Training: Supports training a custom speaker classifier with your own labeled data.
Repository Structure
. ├── speech_model-2-1.py # Main pipeline script (preprocessing, training, inference) ├── sample_data/ # Folder containing training data: organized per speaker ├── pretrained_models/ # Cached pretrained models from SpeechBrain, etc. ├── xgb_classifier.joblib # Trained XGBoost model (generated after training) ├── label_encoder.joblib # Label encoder for speaker labels (generated after training) └── whisperx_diarization_output.xlsx # Output Excel file after diarization and transcription

Setup Instructions
1. Clone the repository
git clone https://github.com/yourusername/your-repo-name.git cd your-repo-name

2. Install dependencies
Use the provided requirements listed below in the Requirements section (you can copy them to a separate requirements.txt if desired), then run pip install -r requirements.txt

Note: Some dependencies are installed from GitHub repositories to get the latest versions (Whisper, WhisperX, Silero VAD).

Usage Guide
Prepare your data
Place your labeled training audio files inside sample_data/

The folder structure must be: sample_data/ ├── speaker_1/ │ ├── file1.wav │ ├── file2.wav ├── speaker_2/ │ ├── file1.wav │ └── ...

Each WAV file should ideally be at least 3 seconds of clean speech from a single speaker.

Run the pipeline
The main script handles all steps — preprocessing, training the speaker classifier, and diarization with transcription:

You'll be prompted to upload test audio files for diarization.
The output Excel file whisperx_diarization_output.xlsx will contain transcriptions segmented by speaker with timestamps and gender info.
Detailed Workflow
Audio Preprocessing: Converts all training and test audio to 16kHz mono WAV.
Embedding Extraction: Uses SpeechBrain's ECAPA-TDNN model to extract speaker embeddings from training data.
Training: Trains an XGBoost classifier on extracted embeddings to distinguish speakers.
Inference: For a new audio file:
Cleans with ffmpeg.
Detects speech segments using Silero VAD.
Performs speaker diarization and embedding extraction.
Predicts speaker labels using the trained classifier.
Transcribes speech segments with WhisperX.
Merges speaker labels and transcription.
Predicts speaker gender from speaker names.
Output: Saves results with speaker names, gender, timestamps, and transcribed texts to an Excel file.
