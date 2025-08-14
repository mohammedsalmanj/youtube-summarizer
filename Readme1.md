# YouTube Video Summarizer & Q&A

This Python project allows users to summarize YouTube videos and ask questions based on the video transcript. It provides concise, unique bullet-point summaries and context-aware answers using modern AI tools.

---

## Features

- Fetch YouTube video transcripts automatically.
- Summarize transcripts into clear and concise bullet points.
- Remove repeated lines and redundant content.
- Ask questions about the video and get context-aware answers.
- Interactive web interface using Gradio.

---

## Technologies and Tools

- Python 3.12+
- Gradio for the user interface
- LangChain for chaining prompts with the language model
- HuggingFace Transformers (FLAN-T5) for natural language processing
- FAISS for semantic vector search in Q&A
- YouTube Transcript API for fetching transcripts
- Utilities for transcript cleaning, chunking, and deduplication


