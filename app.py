# c.py
"""
PDF Chat app — CPU-friendly, robust embedding + chromadb collection usage.
Key improvements for low-RAM CPU:
- Small default batch size (32) and 2 workers
- Use sentence-transformers encode with batching; fallback to HuggingFaceEmbeddings if not present
- Create a fresh chroma collection each process_pdfs run using chromadb.Client().collection.add(...)
- Query by computing query-embedding and calling collection.query(...)
- Avoid any attempt to "patch" embeddings into LangChain internals (prevents misalignment bugs)
Functionality (UI, API) preserved from previous version.
"""
import os
import time
import traceback
import uuid
import gradio as gr
try:
    import gradio_client.utils as gc_utils
    _orig_json = getattr(gc_utils, "json_schema_to_python_type", None)
    _orig__json = getattr(gc_utils, "_json_schema_to_python_type", None)
    _orig_get_type = getattr(gc_utils, "get_type", None)

    def _safe__json_schema_to_python_type(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        if _orig__json is not None:
            return _orig__json(schema, defs)
        return "Any"

    def _safe_json_schema_to_python_type(schema, defs=None):
        try:
            if _orig_json is None:
                return _safe__json_schema_to_python_type(schema, defs)
            return _orig_json(schema, defs)
        except Exception:
            return "Any"

    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        if _orig_get_type is not None:
            return _orig_get_type(schema)
        return "Any"

    gc_utils._json_schema_to_python_type = _safe__json_schema_to_python_type
    gc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type
    gc_utils.get_type = _safe_get_type
except Exception:
    # If patch doesn't apply, continue — gradio may still work.
    pass

# -- Config (tune for your machine) --
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))   # small for 8GB RAM
EMBEDDING_NUM_WORKERS = int(os.environ.get("EMBEDDING_NUM_WORKERS", "2"))
EMBEDDING_NORMALIZE = os.environ.get("EMBEDDING_NORMALIZE", "true").lower() in ("1", "true", "yes")

# -- Optional imports --
try:
    from sentence_transformers import SentenceTransformer
    import torch
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    torch = None
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    _HAS_HF_EMBED = True
except Exception:
    HuggingFaceEmbeddings = None
    _HAS_HF_EMBED = False

try:
    # prefer chromadb client directly for consistent add/query API
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _HAS_CHROMADB = True
except Exception:
    chromadb = None
    ChromaSettings = None
    _HAS_CHROMADB = False

# langchain loader + splitter (for PDF reading & chunking)
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

# genai / gemini client
try:
    from google import genai as genai_pkg
    genai = genai_pkg
except Exception:
    try:
        import genai as genai_pkg
        genai = genai_pkg
    except Exception:
        genai = None

# -- Global state --
_chroma_client = None
_current_collection_name = None
_current_collection = None
_embedding_runner = None  # dict with encode() function
_sentence_model = None

# -- Initialize embedding runner --
def initialize_embeddings(force=False):
    """
    Create an _embedding_runner dict with .encode(texts, batch_size, num_workers) -> numpy array
    Preference: sentence-transformers (fast CPU multi-worker). Fallback: HuggingFaceEmbeddings (slower).
    """
    global _embedding_runner, _sentence_model
    if _embedding_runner is not None and not force:
        return

    if _HAS_SENTENCE_TRANSFORMERS:
        # Use small model by default; CPU will work but keep batches small.
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        try:
            _sentence_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        except Exception:
            # last resort: try default name without path
            _sentence_model = SentenceTransformer(EMBEDDING_MODEL.split("/")[-1], device=device)
        def _encode(texts, batch_size=EMBEDDING_BATCH_SIZE, num_workers=EMBEDDING_NUM_WORKERS):
            # convert_to_numpy returns numpy arrays; normalize if requested
            # show_progress_bar False for background usage
            return _sentence_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=EMBEDDING_NORMALIZE,
                num_workers=num_workers
            )
        _embedding_runner = {"type": "sentence_transformers", "encode": _encode}
        return

    if _HAS_HF_EMBED:
        hf = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        def _hf_encode(texts, batch_size=EMBEDDING_BATCH_SIZE, num_workers=1):
            # HuggingFaceEmbeddings wrapper — process in small batches to limit memory
            import numpy as np
            out = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_emb = hf.embed_documents(batch)
                out.extend(batch_emb)
            return np.array(out)
        _embedding_runner = {"type": "hf", "encode": _hf_encode}
        return

    raise ImportError("No embedding backend available. Install 'sentence-transformers' or 'langchain_huggingface'.")

# -- Chroma client helper (local in-memory) --
def get_chroma_client():
    """
    Return a chromadb.Client instance (in-memory). If chromadb not installed, raise helpful message.
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    if not _HAS_CHROMADB:
        raise ImportError("chromadb not installed. Run: pip install chromadb")
    # Use default settings (in-memory)
    try:
        _chroma_client = chromadb.Client(ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=None))
    except Exception:
        # fallback to default client()
        _chroma_client = chromadb.Client()
    return _chroma_client

# -- Utility: convert LangChain Documents to texts & metadatas --
def docs_to_texts_and_meta(docs):
    texts = []
    metas = []
    for d in docs:
        txt = getattr(d, "page_content", "") or ""
        meta = getattr(d, "metadata", {}) or {}
        # include source hint to help retrieval quality
        src = meta.get("source") or meta.get("filename") or meta.get("path")
        if src:
            txt = f"[source: {src}]\n\n{txt}"
        texts.append(txt)
        metas.append({"source": src} if src else {})
    return texts, metas

# -- Core: process_pdfs (load, split, embed, add to chroma) --
def process_pdfs(files):
    """
    files: list of uploaded file objects from Gradio (each has .name)
    Returns status string.
    """
    global _current_collection_name, _current_collection
    if not files:
        return "Upload at least one PDF."

    if PyPDFLoader is None or RecursiveCharacterTextSplitter is None:
        return "Missing PDF loader or text splitter (install langchain_community and langchain.text_splitter)."

    try:
        initialize_embeddings()
    except Exception as e:
        return f"Embedding init failed: {e}"

    all_docs = []
    try:
        for f in files:
            loader = PyPDFLoader(f.name)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            # ensure all chunks have metadata with source
            for c in chunks:
                if not getattr(c, "metadata", None):
                    c.metadata = {}
                c.metadata.setdefault("source", os.path.basename(f.name))
            all_docs.extend(chunks)
    except Exception as e:
        tb = traceback.format_exc()
        return f"Failed to load/split PDFs: {e}\n{tb}"

    if not all_docs:
        return "No text extracted from PDFs."

    # texts + metadatas
    texts, metadatas = docs_to_texts_and_meta(all_docs)

    # compute embeddings in small batches (safe on 8GB)
    try:
        encode = _embedding_runner["encode"]
        embeddings = encode(texts, batch_size=EMBEDDING_BATCH_SIZE, num_workers=EMBEDDING_NUM_WORKERS)
    except Exception as e:
        # try calling without num_workers (some wrappers have different signatures)
        try:
            embeddings = _embedding_runner["encode"](texts, EMBEDDING_BATCH_SIZE)
        except Exception:
            tb = traceback.format_exc()
            return f"Failed to compute embeddings: {e}\n{tb}"

    # Ensure embeddings is a 2D numpy array-like and length matches texts
    try:
        import numpy as np
        embs = np.asarray(embeddings)
        if embs.shape[0] != len(texts):
            return f"Embedding count mismatch: got {embs.shape[0]} embeddings for {len(texts)} texts."
    except Exception as e:
        return f"Error normalizing embeddings: {e}"

    # Create fresh chroma collection and add data deterministically
    try:
        client = get_chroma_client()
        # create unique collection name (recreate each time to avoid stale vectors)
        name = f"pdf_collection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        # If a collection with the name already exists (unlikely), delete it first
        try:
            _ = client.get_collection(name)
            client.delete_collection(name)
        except Exception:
            pass
        # create collection
        # configure metadata and documents
        collection = client.create_collection(name=name)
        # prepare ids (unique)
        ids = [str(i) for i in range(len(texts))]
        # add to collection (documents are the full text chunks we embed)
        # chromadb expects embeddings as list[list[float]]
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embs.tolist()
        )
        # store global reference
        _current_collection_name = name
        _current_collection = collection
        return f"Processed {len(texts)} chunks into collection '{name}'."
    except Exception as e:
        tb = traceback.format_exc()
        return f"Failed to create chroma collection or add vectors: {e}\n{tb}"

# -- Gemini client helper (same pattern as before) --
def _init_genai_client(api_key_env="GEMINI_API_KEY"):
    if genai is None:
        raise ImportError("GenAI/Gemini SDK not installed. Run: pip install google-genai (or genai).")
    api_key = os.environ.get(api_key_env)
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize genai client: {e}")

DEFAULT_MODEL_FALLBACKS = [
    os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini-2.5-flash-lite",
    "gemini-1.5-mini"
]

def _call_gemini_with_retries(client, model_name, prompt, max_attempts=3, base_delay=1.0):
    import random, time
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if hasattr(response, "text"):
                return True, response.text
            if isinstance(response, dict):
                cands = response.get("candidates") or response.get("generations") or []
                if cands:
                    cand = cands[0]
                    text = cand.get("content", {}).get("text") or cand.get("text")
                    if text:
                        return True, text
            return True, str(response)
        except Exception as e:
            err = str(e)
            if ("429" in err) or ("rate limit" in err.lower()):
                delay = base_delay * (2 ** (attempt - 1)) + random.random() * 0.2
                time.sleep(delay)
                continue
            return False, f"Gemini error: {err}"
    return False, f"Rate-limited after {max_attempts} attempts for model {model_name}"

# -- Chat: compute query embedding, query chroma collection, build prompt, call Gemini --
def chat_response(message, history):
    """
    Returns text answer. Requires process_pdfs to have been called.
    """
    global _current_collection
    if _current_collection is None:
        return "Please upload and process PDF documents first."

    try:
        initialize_embeddings()
    except Exception as e:
        return f"Embeddings initialization error: {e}"

    try:
        # create query embedding
        encode = _embedding_runner["encode"]
        try:
            q_emb = encode([message], batch_size=EMBEDDING_BATCH_SIZE, num_workers=EMBEDDING_NUM_WORKERS)[0].tolist()
        except Exception:
            q_emb = encode([message], EMBEDDING_BATCH_SIZE)[0].tolist()

        # query the collection
        # request top k documents; tune k=3
        query_res = _current_collection.query(query_embeddings=[q_emb], n_results=5, include=["documents", "metadatas", "distances"])
        documents = []
        if query_res and "documents" in query_res and len(query_res["documents"]) > 0:
            for doc_list in query_res["documents"]:
                for doc_text in (doc_list or []):
                    documents.append(doc_text)
        # if none found, return not-found message
        if not documents:
            return "I cannot find that information in the provided documents."

        # short-circuit: build context of the top results (limit length)
        context = "\n\n---\n\n".join([d[:1500] for d in documents])

        prompt = (
            "You are a very very very helpful assistant. Use ONLY the CONTEXT below to answer the question in great details and accuracy to the point dear. Also use your model power to give extra informations about the question.\n\n"
            "CONTEXT:\n" + context + "\n\n"
            "QUESTION:\n" + message + "\n\n"
            "INSTRUCTIONS:\n- Answer concisely and accurately using only the context, but in full great details and to the point. Also use your model power to give extra detailed informations about the question."
            "If the answer is not present in the context, reply: 'Ayhayy kichui pailam na dekhi!!!'"
        )

        # call Gemini with fallbacks
        try:
            client = _init_genai_client()
        except Exception as e:
            return f"Failed to init Gemini client: {e}"

        last_err = None
        for model_name in DEFAULT_MODEL_FALLBACKS:
            ok, res = _call_gemini_with_retries(client, model_name, prompt, max_attempts=3, base_delay=1.0)
            if ok:
                return res
            last_err = res

        return f"All model attempts failed: {last_err}"

    except Exception as e:
        tb = traceback.format_exc()
        return f"chat_response error: {e}\n{tb}"
    
with gr.Blocks(title="PDF-Chat") as demo:

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDFs")
            process_btn = gr.Button("Process PDFs", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=True)

        with gr.Column(scale=2):
            if hasattr(gr, "ChatInterface"):
                try:
                    gr.ChatInterface(fn=chat_response,
                                     examples=["Summarize the document like an expert.", "What are the algorithms mentioned?", "What are the equations?"],
                                     title="Chatbot",
                                     description="Ask questions about uploaded PDFs.")
                except Exception:
                    chat_display = gr.Chatbot(label="Chatbot")
                    user_input = gr.Textbox(show_label=False, placeholder="Type your question here...(Be specific and concise to get better answers.)")
                    submit_btn = gr.Button("Send")
                    def _wrapped_chat(input_text, history):
                        answer = chat_response(input_text, history)
                        history = history or []
                        history.append((input_text, answer))
                        return history, ""
                    submit_btn.click(_wrapped_chat, inputs=[user_input, chat_display], outputs=[chat_display, user_input])
            else:
                chat_display = gr.Chatbot(label="Chatbot")
                user_input = gr.Textbox(show_label=False, placeholder="Type your question here...(Be specific and concise to get better answers.)")
                submit_btn = gr.Button("Send")
                def _wrapped_chat(input_text, history):
                    answer = chat_response(input_text, history)
                    history = history or []
                    history.append((input_text, answer))
                    return history, ""
                submit_btn.click(_wrapped_chat, inputs=[user_input, chat_display], outputs=[chat_display, user_input])

    process_btn.click(fn=process_pdfs, inputs=[file_input], outputs=[status_output])

# Launch (adjust share/host args as you need)
if __name__ == "__main__":
    demo.launch(share=True)