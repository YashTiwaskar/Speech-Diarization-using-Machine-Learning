# Speech Diarization

This repository provides a comprehensive pipeline for speech diarization, speaker recognition, and speech transcription using state-of-the-art models such as WhisperX, Silero VAD, and SpeechBrain. It processes audio files, performs voice activity detection, transcribes speech with speaker labels, predicts gender, and exports the diarized results to an Excel file.

The system uses advanced machine learning techniques, including voice activity detection (VAD), feature extraction, speaker embedding, and clustering algorithms to accurately distinguish multiple speakers in an audio stream.

# Features

Audio Preprocessing: Converts input audio to 16kHz mono WAV automatically.
Voice Activity Detection (VAD): Uses Silero VAD to detect speech segments.
Speech Transcription: High-quality transcription using WhisperX with automatic language detection.
Speaker Diarization: Speaker segmentation and labeling via SpeechBrain embeddings and a trained XGBoost classifier.
Gender Prediction: Infers speaker gender (male, female, unknown) using the speaker name.
Output: Saves diarized transcripts with speaker and gender info into an Excel .xlsx file.
Custom Training: Supports training a custom speaker classifier with your own labeled data.
