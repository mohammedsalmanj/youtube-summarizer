
## Usage

1. Start the application:
2. Open the Gradio interface in your browser (or via the shared link if `share=True` is enabled).

3. Steps in the interface:

   - Enter a YouTube URL to fetch and summarize the video.
   - View the bullet-point summary.
   - Ask questions based on the transcript and receive context-aware answers.

---

## How It Works

1. Transcript Handling  
   - Fetch transcript using the YouTube Transcript API.
   - Phase 2 includes cleaning and removing repeated lines.

2. Chunking  
   - Split transcript into manageable chunks for the language model.
   - Clean chunks improve summarization and Q&A accuracy.

3. Prompt Design  
   - Phase 1 used a simple "summarize this" prompt.
   - Phase 2 includes detailed instructions for unique bullet points and key facts.

4. Summary Post-processing  
   - Phase 1 had no post-processing.
   - Phase 2 deduplicates repeated lines to improve readability.

5. FAISS for Q&A  
   - Phase 1 stored raw chunks with redundant information.
   - Phase 2 stores cleaned chunks for more precise context-based answers.

---

## Phase-wise Improvements

| Feature / Aspect            | Phase 1 (V1)                                         | Phase 2 (V2)                                                              | Analysis / Improvement                                                           |
| --------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Transcript Handling          | Raw transcript from YouTube                          | Cleaned transcript removing repeated words and lines                       | Reduced redundancy and improved summary quality                                  |
| Chunking                     | Fixed-size chunks, naive                             | Same chunks but from cleaned transcript                                    | Cleaner chunks lead to better summarization and Q&A context                      |
| Prompt Design                | Simple summarize instruction                         | Detailed prompt with unique bullet points and focus on key facts           | More structured and precise outputs                                              |
| Summary Post-processing      | None                                                 | Deduplication of repeated lines                                           | Avoids repetition and improves readability                                       |
| FAISS / Q&A                  | Raw chunks with redundancy                           | Cleaned chunks for indexing                                               | Context for Q&A is precise, answers less repetitive                               |
| Output Quality               | Repetitive summaries                                 | Clear, concise, and non-redundant bullet points                            | Phase 2 output is professional and usable                                        |
| Challenges Faced             | Repeated phrases, messy output, poor readability    | Cleaning and deduplication reduced issues                                  | Future improvements can include semantic chunking and hierarchical summarization |

---

## Future Improvements

- Support for longer videos using hierarchical chunking.
- Advanced semantic Q&A using vector databases like Pinecone.
- Save summaries and Q&A sessions locally for later reference.
- Add multilingual support for non-English videos.

---

