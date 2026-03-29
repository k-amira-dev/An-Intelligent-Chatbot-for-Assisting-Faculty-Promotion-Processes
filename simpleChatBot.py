import streamlit as st
import os
import faiss
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# --- PROMPTS ---
uos_qa_template = PromptTemplate(
    "You are an expert Faculty Affairs Consultant at the University of Sharjah.\n"
    "Context:\n{context_str}\n"
    "Using ONLY the context, answer: {query_str}\n"
    "If the answer is not in the context, say 'Not found.'\n"
    "Use a formal, helpful tone.\n"
    "Answer: "
)

# --- SETTINGS ---
def set_llm():
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_KEY"])
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=100)

@st.cache_resource 
def load_data_from_github():
    if not os.path.exists("data"): return None
    parser = LlamaParse(api_key=st.secrets["LLAMA_CLOUD_API_KEY"], result_type="markdown")
    reader = SimpleDirectoryReader("data", file_extractor={".pdf": parser})
    docs = reader.load_data()

    # FAISS Setup (Matches your IndexFlatIP logic)
    d = 384 
    faiss_index = faiss.IndexFlatIP(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# --- HYBRID RETRIEVER (REPLICATING YOUR LOGIC) ---
class CustomHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle):
        # 1. Semantic Search (FAISS)
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        # 2. Keyword Search (BM25)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        # 3. Combine & Deduplicate (Using text content as the key)
        all_nodes = vector_nodes + bm25_nodes
        unique_nodes_dict = {n.node.get_content(): n for n in all_nodes}
        
        return list(unique_nodes_dict.values())

def get_query_engine(index):
    # Setup base retrievers
    vector_retriever = index.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, 
        similarity_top_k=5
    )

    hybrid_retriever = CustomHybridRetriever(vector_retriever, bm25_retriever)
# 4. NEW: Setup the LLM Reranker
    # choice_batch_size: How many chunks it looks at once
    # top_n: How many total chunks it passes to the final answer
    reranker = LLMRerank(choice_batch_size=5, top_n=3)

    # 5. Build the engine and plug in the reranker
    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],  #the reranker is plugged in here!!
        text_qa_template=uos_qa_template
    )
    # Returned the query engine using qa uos template
   
def evaluate_response(query, response_text, context_nodes):
    # Extract text from the source nodes retrieved by the hybrid search
    context_text = "\n".join([n.get_content() for n in context_nodes])
    
    # Your updated detailed Auditor Prompt
    judge_prompt = f"""
    ROLE: Senior Academic Auditor
    TASK: Verify the accuracy of a generated answer against the provided Source Context.
    
    SOURCE CONTEXT: 
    {context_text}
    
    USER QUERY: {query}
    GENERATED ANSWER: {response_text}
    
    AUDIT REQUIREMENTS:
    1. FACTUALITY: Does the answer contain ANY information not present in the Source Context?
    2. CITATION: Does the answer correctly cite Article numbers if they exist?
    3. HALLUCINATION: Did the AI make up any dates, numbers, or rules that don't exist in the documents? 
    4. KEYWORDS: Does the generated answer contain keywords that exist in the articles?
    
    FINAL VERDICT:
    - Confidence Score: (1 to 5)
    - Hallucination Detected: (Yes/No)
    - Reasoning: (Briefly explain why).
    """
    
    # Use the global LLM (Groq) to perform the audit
    return str(Settings.llm.complete(judge_prompt))

#shorter version if you want to use it amira
#def evaluate_response(query, response_text, context_nodes):
 #   context_text = "\n".join([n.get_content() for n in context_nodes])
  #  judge_prompt = f"Verify answer against context:\n{context_text}\nQuery: {query}\nAnswer: {response_text}"
   # return str(Settings.llm.complete(judge_prompt))
