import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.readers.file import PDFReader
from llama_index.retrievers.bm25 import BM25Retriever

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextRecall, Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

DOCS_PATH = (
    "C:/Users/amour/OneDrive - University of Sharjah/"
    "amira 3rd year/spring/junior project/promotion documents"
)
GT_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
GT_THRESHOLD = 0.85
MIN_SIMILARITY = 0.5
TOP_K = 3

# ── Pydantic Models ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class EvaluationScores(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float] = None

class SourceRef(BaseModel):
    file_name: str
    page: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: list[SourceRef]
    evaluation: EvaluationScores
    ground_truth_used: bool = False

# ── Prompt ─────────────────────────────────────────────────────────────────────

QA_TEMPLATE = PromptTemplate(
    "You are an expert Faculty Affairs Consultant at the University of Sharjah.\n\n"
    "Answer the question using ONLY the provided context. Follow these rules:\n"
    "- Give a comprehensive, well-structured answer with all relevant details.\n"
    "- For planning or step-by-step questions, organize your answer with numbered steps.\n"
    "- For requirement questions, list all conditions and criteria mentioned in the context.\n"
    "- If the topic is completely absent from the context, respond with: Not found.\n"
    "- Do NOT use general knowledge not present in the context.\n\n"
    "Context:\n{context_str}\n\n"
    "Question:\n{query_str}\n\n"
    "Answer:"
)

# ── Global State ────────────────────────────────────────────────────────────────

query_engine = None
gt_entries = []
gt_embeddings = None

# ── Startup ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_):
    global query_engine
    logger.info("Initializing system...")

    Settings.llm = LlamaGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=100)

    docs = SimpleDirectoryReader(DOCS_PATH, file_extractor={".pdf": PDFReader()}).load_data()
    index = VectorStoreIndex.from_documents(docs)

    vector_ret = index.as_retriever(similarity_top_k=10)
    bm25_ret = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
    fusion = QueryFusionRetriever(
        [vector_ret, bm25_ret], similarity_top_k=10, num_queries=1,
        mode="reciprocal_rerank", use_async=True
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion,
        node_postprocessors=[LLMRerank(choice_batch_size=5, top_n=TOP_K)],
        text_qa_template=QA_TEMPLATE
    )

    load_ground_truth()
    logger.info("System ready.")
    yield

# ── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(title="UoS Faculty Affairs Chatbot", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ── Ground Truth ────────────────────────────────────────────────────────────────

def load_ground_truth():
    global gt_entries, gt_embeddings
    if not os.path.exists(GT_PATH):
        logger.warning("ground_truth.json not found — context_recall will be skipped.")
        return
    with open(GT_PATH, encoding="utf-8") as f:
        gt_entries = json.load(f)
    gt_embeddings = np.array(
        [Settings.embed_model.get_text_embedding(e["question"]) for e in gt_entries],
        dtype=np.float32
    )
    logger.info("Loaded %d ground truth entries.", len(gt_entries))

def find_ground_truth(query: str) -> Optional[str]:
    if not gt_entries or gt_embeddings is None:
        return None
    q_emb = np.array(Settings.embed_model.get_text_embedding(query), dtype=np.float32)
    norms = np.linalg.norm(gt_embeddings, axis=1) * np.linalg.norm(q_emb)
    sims = np.dot(gt_embeddings, q_emb) / np.maximum(norms, 1e-9)
    best = int(np.argmax(sims))
    return gt_entries[best]["answer"] if sims[best] >= GT_THRESHOLD else None

# ── RAGAS Evaluation ────────────────────────────────────────────────────────────

def run_evaluation(question: str, answer: str, contexts: list[str], ground_truth: Optional[str]) -> EvaluationScores:
    f_m = Faithfulness()
    r_m = ResponseRelevancy()
    p_m = LLMContextPrecisionWithoutReference()
    metrics = [f_m, r_m, p_m]

    rc_m = None
    if ground_truth:
        rc_m = ContextRecall()
        metrics.append(rc_m)

    sample = SingleTurnSample(
        user_input=question, response=answer,
        retrieved_contexts=contexts, reference=ground_truth
    )
    scores = evaluate(EvaluationDataset(samples=[sample]), metrics=metrics).to_pandas().iloc[0]

    def get(m):
        v = scores.get(m.name, 0.0)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0

    return EvaluationScores(
        faithfulness=get(f_m),
        answer_relevancy=get(r_m),
        context_precision=get(p_m),
        context_recall=get(rc_m) if rc_m else None,
    )

# ── Helpers ─────────────────────────────────────────────────────────────────────

def context_is_on_topic(query: str, context: str) -> bool:
    """Checks if the retrieved context is about the same topic as the query — not whether it answers it exactly."""
    res = Groq(api_key=os.getenv("GROQ_API_KEY")).chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content":
            f"Does the context below contain information related to the topic of the question?\n"
            f"Answer YES or NO only.\n\nQuestion: {query}\n\nContext: {context}"
        }],
        temperature=0,
    )
    return "YES" in res.choices[0].message.content.upper()

def extract_sources(nodes) -> list[SourceRef]:
    seen, sources = set(), []
    for node in nodes:
        file_name = node.node.metadata.get("file_name", "Unknown")
        page = node.node.metadata.get("page_label")
        if page and (file_name, page) not in seen:
            seen.add((file_name, page))
            sources.append(SourceRef(file_name=file_name, page=str(page)))
    return sources

def not_found() -> ChatResponse:
    return ChatResponse(
        response="Not found.",
        sources=[],
        evaluation=EvaluationScores(faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0),
    )

# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "ready": query_engine is not None}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not query_engine:
        raise HTTPException(503, "System not ready.")
    if not request.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    try:
        response = query_engine.query(request.message)
    except Exception as e:
        logger.exception("Query error: %s", e)
        raise HTTPException(500, "Query failed.")

    if not response.source_nodes:
        return not_found()

    if (response.source_nodes[0].score or 0) < MIN_SIMILARITY:
        return not_found()

    context_text = "\n".join(n.node.text for n in response.source_nodes[:TOP_K])
    if not context_is_on_topic(request.message, context_text):
        return not_found()

    answer = str(response)
    if "not found" in answer.lower():
        return not_found()

    ground_truth = find_ground_truth(request.message)
    contexts = [n.node.text for n in response.source_nodes]

    try:
        evaluation = run_evaluation(request.message, answer, contexts, ground_truth)
    except Exception as e:
        logger.warning("RAGAS evaluation failed: %s", e)
        evaluation = EvaluationScores(faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0)

    return ChatResponse(
        response=answer,
        sources=extract_sources(response.source_nodes),
        evaluation=evaluation,
        ground_truth_used=ground_truth is not None,
    )
