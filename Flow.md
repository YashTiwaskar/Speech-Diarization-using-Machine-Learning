

## **Project Flow Overview**

1. **Audio Data Preparation**
2. **Voice Activity Detection (VAD)**
3. **Training the Speaker Classifier**
4. **Transcription \& Alignment**
5. **Speaker Diarization (Speaker Segmentation)**
6. **Speaker Labeling \& Gender Detection**
7. **Result Merging and Output (Excel Export)**
8. **Deployment / Usability**

## Key Libraries/Modules and Their Roles

| Library/Module               | Purpose                                               |
|------------------------------|------------------------------------------------------|
| ffmpeg-python                | Audio format conversion and normalization            |
| torchaudio + torch           | Audio loading, resampling, segmenting                |
| speechbrain                  | Speaker embedding extraction (ECAPA-TDNN model)      |
| xgboost (XGBClassifier)      | Speaker label classification                         |
| sklearn LabelEncoder         | Encode/decode speaker name <–> integer mapping       |
| joblib                       | Store/fetch trained models and encoders              |
| whisperx                     | Transcription, alignment, diarization (ASR) pipeline |
| pyannote.audio (via whisperx)| Speaker diarization backend                          |
| gender-guesser               | Guess gender from predicted speaker name             |
| pandas                       | Data processing, tabular structuring, Excel export   |
| BytesIO                      | In-memory transfer for downloadable Excel            |


## **Detailed Stage-by-Stage Explanation**

### **1. Audio Data Preparation**

- **Goal:** Ensure all audio files are in a consistent, ML-friendly format (16kHz mono WAV).
- **How:** Use `torchaudio` to:
    - Find all `.wav` files.
    - Convert stereo files to mono (average channels).
    - Resample any non-16kHz to 16kHz.
    - Save back (overwrite).

**Why:** Models require fixed sample rate and channel count for accuracy and reproducibility.

### **2. Voice Activity Detection (VAD)**

- **Goal:** Split long audio into smaller speech segments, removing silence/background.
- **How:**
    - Load the Silero VAD model (via TorchHub).
    - Apply to cleaned audio to get time-stamped segments where speech is detected.

**Why:** Only feed actual speech to downstream diarization and transcription for speed and performance.

### **3. Training the Speaker Classifier**

- **Goal:** Recognize/label speakers based on their voice.
- **How:**
    - For each speaker (each folder in training ZIP), extract 3-second chunks from their audio.
    - Use SpeechBrain’s ECAPA-TDNN to extract speaker embeddings from each chunk—these are high-level vector representations of voices.
    - For each chunk, label with the speaker’s name (from the folder).
    - Train an **XGBoost classifier** on these embeddings and the corresponding labels.
    - Save the classifier (`xgb_classifier.joblib`) and label encoder (`label_encoder.joblib`) for prediction later.

**Why:** Enables personalized speaker labeling, not just generic “Speaker 1, Speaker 2.”

### **4. Transcription \& Word Alignment**

- **Goal:** Convert speech into text with precise timestamps.
- **How:**
    - Use **WhisperX** for fast, accurate ASR (Automatic Speech Recognition).
    - WhisperX `transcribe()`:
        - Detects language (e.g. Hindi, English, etc.)—outputs `"language"` field.
        - Transcribes speech to text with segment start/end times.
    - WhisperX alignment step (`align()`) aligns this text at the word level for greater timing resolution.

**Why:** Knowing what was said and when is essential for diarization and later speaker-to-text matching.

### **5. Speaker Diarization**

- **Goal:** Segment audio by who is speaking, i.e., “diarize” by speaker turns.
- **How:**
    - Use **WhisperX’s DiarizationPipeline** (backed by `pyannote.audio`, requisites handled via Hugging Face PyPI and access token).
    - Produces, for the input audio, start/end times of continuous same-speaker regions.

**Why:** Allows you to know **when** each person speaks, not just what is said.

### **6. Speaker Labeling \& Gender Detection**

- **Goal:** Assign names (not just “Speaker N”) and detect gender for each diarized segment.
- **How:**
    - For each detected diarization segment:
        - Extract the corresponding audio chunk.
        - Compute the speaker embedding (via SpeechBrain ECAPA-TDNN).
        - Predict speaker using your trained XGBoost model.
        - Use the `gender-guesser` library to assign a likely gender to each speaker label.

**Why:** Having true *identity* and gender linked to every spoken segment makes transcripts and reports far more actionable.

### **7. Merging Results \& Exporting**

- **Goal:** Combine all information for final output.
- **How:**
    - Merge aligned transcripts with diarization windows (by closest/overlapping times).
    - For each merged segment, build a row with start/end times, predicted speaker, and gender (optional: file name).
    - Export results to Excel (`whisperx_diarization_output.xlsx`) using `pandas`.

**Why:** Enables post-analysis and user-friendly review or sharing.

### **8. Deployment / Usability (through Streamlit or Scripts)**

- Allow user to select audio files/ZIP, train the model in-app, and get diarization/excel output interactively.
- Designed for flexibility: can be run in Jupyter, as a script, or as a forthcoming Streamlit app with file upload/download buttons.


## **Flow Diagram**

**Preprocessing** ⟶ **VAD** ⟶ **Extract speaker embeddings** ⟶ **Train XGBoost speaker classifier**
          ⬇
**User audio for diarization** ⟶ **Preprocessing** ⟶ **Diarization** + **Transcription**
             ⬇
**Speaker embeddings for segments** ⟶ **Speaker label prediction**
             ⬇
**(Optional: Gender detection)**
             ⬇
**Merge with transcript** ⟶ **Excel export**

## **Key Points for Presentation**

- **Your system does:**
    - Custom speaker recognition (not just anonymous diarization)
    - High-accuracy ASR with word-level timing
    - End-to-end process: pre-processing, training, inference, merging, export
- **Models used:**
    - WhisperX for transcription and diarization
    - SpeechBrain ECAPA-TDNN for voice embeddings
    - XGBoost for speaker classification
    - gender-guesser for automatic gender attribution
- **Why it matters:**
 Safer, more actionable meeting analysis, call-center/HR/compliance tools, teaching, media annotation, etc.


## **Suggested Slide Structure**

1. **Objective \& Motivation**
2. **System Architecture/Flow Diagram**
3. **Key Steps (with screenshots/visual snippets if possible)**
    - Preprocessing, Diarization, Embeddings, Classifier Training, Diarization+Transcribe, Speaker/Gender Attribution, Export
4. **Sample Outputs**
    - Show an Excel output (with columns: filename, start, end, speaker, gender)
    - Speaker timeline chart or table
5. **Technical Stack \& Models Used**
6. **Challenges \& Solutions**
7. **Applications / Future Work**
