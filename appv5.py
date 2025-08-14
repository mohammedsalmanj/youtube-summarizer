import gradio as gr
from utils import get_video_id, fetch_transcript, chunk_text, create_faiss_index, remove_repeated_lines
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ------------------ LLM Setup ------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# ------------------ Prompts ------------------
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the transcript into clear, concise bullet points.
- Each point must be unique.
- Remove repeated words and sentences.
- Focus on key facts only.

Transcript:
{text}
"""
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the context to answer the question:\nContext: {context}\nQuestion: {question}"
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# ------------------ Global Storage ------------------
faiss_index = None
full_transcript = ""

# ------------------ Helper Functions ------------------
def deduplicate_summary(summary):
    """Remove repeated lines from summary."""
    lines = summary.split("\n")
    seen = set()
    unique_lines = []
    for line in lines:
        clean_line = line.strip()
        if clean_line and clean_line not in seen:
            unique_lines.append(clean_line)
            seen.add(clean_line)
    return "\n".join(unique_lines)

# ------------------ Gradio Functions ------------------
def summarize_video(url):
    global full_transcript
    video_id = get_video_id(url)
    if not video_id:
        return "Invalid URL"
    try:
        full_transcript = fetch_transcript(video_id)
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

    # ------------------ Clean transcript ------------------
    full_transcript_clean = remove_repeated_lines(full_transcript)

    # ------------------ Chunk and Summarize ------------------
    chunks = chunk_text(full_transcript_clean)
    summaries = [summary_chain.run({"text": chunk["page_content"]}) for chunk in chunks]

    # First-level combined summary
    combined_summary = " ".join(summaries)

    # Second-level bullet-point summary
    final_summary = summary_chain.run({"text": combined_summary})

    # Post-process summary
    final_summary = deduplicate_summary(final_summary)

    return final_summary.strip()

def answer_question(question):
    global faiss_index, full_transcript
    if not full_transcript:
        return "Fetch a video first!"
    if faiss_index is None:
        chunks = chunk_text(full_transcript)
        faiss_index = create_faiss_index(chunks)
    docs = faiss_index.similarity_search(question, k=3)
    context = " ".join([d.page_content for d in docs])
    return qa_chain.run({"context": context, "question": question})

# ------------------ Gradio Interface ------------------
with gr.Blocks() as interface:
    gr.Markdown("# YouTube Video Summarizer & Q&A")

    url_input = gr.Textbox(label="YouTube URL")
    summary_output = gr.Textbox(label="Video Summary", lines=10)
    summarize_btn = gr.Button("Summarize Video")
    summarize_btn.click(summarize_video, inputs=url_input, outputs=summary_output)

    question_input = gr.Textbox(label="Ask a Question")
    answer_output = gr.Textbox(label="Answer", lines=5)
    question_btn = gr.Button("Get Answer")
    question_btn.click(answer_question, inputs=question_input, outputs=answer_output)

interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
