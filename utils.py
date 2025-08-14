import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
from faster_whisper import WhisperModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ YouTube Video Helpers ------------------

def get_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_transcript(video_id: str) -> str:
    """Fetch transcript using YouTubeTranscriptApi; fallback to Whisper if unavailable."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([t['text'] for t in transcript_list])
        transcript = remove_repeated_words(transcript)
        transcript = remove_repeated_phrases(transcript)
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound, AttributeError):
        return transcribe_with_whisper(video_id)

def transcribe_with_whisper(video_id: str) -> str:
    """Download video audio using yt-dlp and transcribe with Whisper."""
    url = f"https://youtu.be/{video_id}"
    ydl_opts = {"format": "bestaudio/best", "outtmpl": f"{video_id}.%(ext)s"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info)

    model = WhisperModel("small")
    segments, _ = model.transcribe(audio_file)
    transcript = " ".join([seg.text for seg in segments])
    transcript = remove_repeated_words(transcript)
    transcript = remove_repeated_phrases(transcript)
    return transcript

# ------------------ FAISS Helpers ------------------

def chunk_text(text: str, chunk_size: int = 150, chunk_overlap: int = 30) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({"page_content": chunk})
        start += chunk_size - chunk_overlap
    return chunks

def create_faiss_index(chunks: list):
    """Create FAISS index from text chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings()
    texts = [c["page_content"] for c in chunks]
    return FAISS.from_texts(texts, embeddings)

# ------------------ Text Cleaning Helpers ------------------

def remove_repeated_words(text: str) -> str:
    """Replace consecutive repeated words with a single instance."""
    words = text.split()
    if not words:
        return ""
    cleaned = [words[0]]
    for i in range(1, len(words)):
        if words[i] != words[i-1]:
            cleaned.append(words[i])
    return " ".join(cleaned)

def remove_repeated_lines(text: str) -> str:
    """Remove repeated sentences in a transcript."""
    lines = text.split(". ")
    seen = set()
    unique_lines = []
    for line in lines:
        clean_line = line.strip()
        if clean_line and clean_line not in seen:
            unique_lines.append(clean_line)
            seen.add(clean_line)
    return ". ".join(unique_lines)

def remove_repeated_phrases(text: str) -> str:
    """
    Removes consecutive repeated phrases of 2+ words, including punctuation.
    Example: 'Kubernetes vs. Kubernetes: Kubernetes vs. Kubernetes:' -> 'Kubernetes vs.'
    """
    pattern = r'\b(\w+(?: \w+){0,5})\b(?:\s*\1\b)+'
    return re.sub(pattern, r'\1', text, flags=re.IGNORECASE)

def deduplicate_summary(summary: str) -> str:
    """Removes repeated lines and repeated phrases in summary."""
    summary = remove_repeated_phrases(summary)
    lines = summary.split("\n")
    seen = set()
    unique_lines = []
    for line in lines:
        line_clean = line.strip()
        if line_clean and line_clean not in seen:
            unique_lines.append(line_clean)
            seen.add(line_clean)
    return "\n".join(unique_lines)
