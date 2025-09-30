# AI Voice Assistant: Speech Translation, Transcription and Conversational Interaction

This project contains an AI-driven voice assistant that integrates **ASR**, **speech-to-text translation** and **LLM Interaction** with a unified interface. It leverages **Whisper** for transcription and translation jobs and employs **Qwen** for natural language dialgoue. The interface is deployed using **Streamlit**, enbaling accessibility through an interactive and simple web application.

## Abstract
The goal is to design and implement a modular voice assistantthat performs three primary tasks:
1. Transcription of spoken language into text.
2. Translation of spoken language into English.
3. Conversational chatbot through LLM.

## Features
- **Audio Input** : Users may directly record an audio or upload a pre-recorded audio file through the interface.
- **ASR**: Transcription of input audio using **OpenAI Whisper**.
- **Speech Translation**: Cross-lingual translation to English.
- **Conversational Agent**: Dialogue with **Qwen** configured as a responsive and contextaware assistant.
- **User Interface**: Web-based development using **Streamlit**.

## Implementation
- **Language** : Python 3.9>
- **Framework**: Streamlit
- **Models**:
  - whisper-small
  - Qwen-3 0.6B
- **Libraries**:
  - `transformers`
  - `torch`
  - `streamlit`
  - `tempfile` (for audio handling)

  
