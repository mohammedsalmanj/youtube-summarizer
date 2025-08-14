import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
from faster_whisper import WhisperModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ YouTube Video Helpers ------------------

def get_video_id(url: str) -> str:
    """
    Extract the video ID from a YouTube URL.
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript using YouTubeTranscriptApi; if unavailable, fallback to Whisper.
    """
    try:
        # Try official YouTube transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([t['text'] for t in transcript_list])
    except (TranscriptsDisabled, NoTranscriptFound, AttributeError):
        # Fallback: download audio + transcribe with Whisper
        return transcribe_with_whisper(video_id)

def transcribe_with_whisper(video_id: str) -> str:
    """
    Download video audio using yt-dlp and transcribe with Whisper.
    """
    url = f"https://youtu.be/{video_id}"
    ydl_opts = {"format": "bestaudio/best", "outtmpl": f"{video_id}.%(ext)s"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info)

    # Load Whisper model
    model = WhisperModel("small")  # can switch to medium/large
    segments, _ = model.transcribe(audio_file)
    transcript = " ".join([seg.text for seg in segments])
    return transcript

# ------------------ FAISS Helpers ------------------

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({"page_content": chunk})
        start += chunk_size - chunk_overlap
    return chunks

def create_faiss_index(chunks: list):
    """
    Create FAISS index from text chunks using HuggingFace embeddings.
    """
    embeddings = HuggingFaceEmbeddings()
    texts = [c["page_content"] for c in chunks]
    return FAISS.from_texts(texts, embeddings)
