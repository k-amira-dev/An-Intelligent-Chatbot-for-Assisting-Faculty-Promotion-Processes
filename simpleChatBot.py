# import libraries

import os  #to read files
import pdfplumber   #to extract text from PDF files
import faiss  # Facebook AI Similarity Search to store and search vector embeddings efficiently
import tiktoken # to convert text into tokens
from sentence_transformers import SentenceTransformer # to convert text into vector embeddings
from groq import Groq # cloud service to access LLM model
from rank_bm25 import BM25Okapi # Keyword retrieval

# create a Groq client using the API key stored as an environment variable
# This allows the program to communicate with the LLM model hosted by Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#function to extract text from pdf
def extract_text_from_pdfs(folder_path):

    # List that will store each document with its source name and extracted text
    documents = []

    # Loop through all files inside the folder
    for filename in os.listdir(folder_path):

        # Process only PDF files
        if filename.endswith(".pdf"):

            # Build the full file path
            filepath = os.path.join(folder_path, filename)
            print(f"Reading {filename}...")

            # Open the PDF file using pdfplumber
            with pdfplumber.open(filepath) as pdf:
                full_text = ""

                # Loop through every page of the PDF
                for page in pdf.pages:
                    # Extract text from the page
                    text = page.extract_text()
                    # Only add text if extraction succeeded
                    if text:
                        full_text += text + "\n"

            # Store the document text and its source filename
            documents.append({
                "source": filename,
                "text": full_text
            })
    # Return the list of extracted documents
    return documents

#function to clean text
def clean_text(text):

    # Replace newline characters with spaces
    text = text.replace("\n", " ")
    # Remove extra spaces by splitting and joining
    text = " ".join(text.split())

    return text

#function to chunk text into smaller parts
def chunk_text(text, source, chunk_size=400, overlap=100):

    # Load tokenizer used by many LLM models
    enc = tiktoken.get_encoding("cl100k_base")
    # Convert text into tokens
    tokens = enc.encode(text)

    chunks = []
    start = 0

    # Loop until the entire text is chunked
    while start < len(tokens):
        # Define the end of the chunk
        end = start + chunk_size
        # Extract token segment
        chunk_tokens = tokens[start:end]
        # Convert tokens back into readable text
        chunk_string = enc.decode(chunk_tokens)
        # Store the chunk along with its source file
        chunks.append({
            "text": chunk_string,
            "source": source
        })

        # Move start position forward but keep overlap
        start += chunk_size - overlap

    return chunks

#function to uild FAISS vector index
def build_index(chunks, model):

    # Extract only the text content of chunks
    texts = [chunk["text"] for chunk in chunks]
    # Convert each chunk into vector embeddings
    embeddings = model.encode(texts).astype("float32")
    # Normalize vectors so cosine similarity can be used
    faiss.normalize_L2(embeddings)
    # Create FAISS index using inner product (cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    # Add embeddings to the index
    index.add(embeddings)

    return index

#function to retrieve most relevant chunks: Hybrid Retrieval (Semantic + Keyword)
def retrieve(query, chunks, index, model, bm25, k=10):

    # Instruction improves embedding quality for search
    query_instruction = "Represent this sentence for searching relevant passages: "
    # Convert query into vector embedding
    q_vec = model.encode([query_instruction + query]).astype("float32")
    # Normalize query vector
    faiss.normalize_L2(q_vec)
    # Search FAISS index for top k most similar vectors
    D, I = index.search(q_vec, k)
    # Retrieve semantic search results
    semantic_chunks = [chunks[i] for i in I[0]]

    # Tokenize query for BM25 keyword search
    tokenized_query = query.lower().split()
    # Compute BM25 relevance scores
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get indices of top k keyword matches
    top_keyword_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    # Retrieve keyword search results
    keyword_chunks = [chunks[i] for i in top_keyword_indices]

    # Merge semantic and keyword results
    combined = semantic_chunks + keyword_chunks

    # Remove duplicates using dictionary
    unique = {chunk["text"]: chunk for chunk in combined}

    return list(unique.values())

# function to generate answers using LLM
def generate_answer(query, retrieved_chunks):

    context = ""
    sources = set()

    # Combine retrieved chunks into a single context
    for chunk in retrieved_chunks:
        context += chunk["text"] + "\n\n"
        # Track document sources
        sources.add(chunk["source"])

    # Send prompt to the LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[

            # System instruction that controls model behavior
            {
                "role": "system",
                "content": "You are an assistant helping university faculty understand promotion requirements."
                           "Use ONLY the provided context."
                           "If the question asks for guidance or a plan, synthesize the answer using the requirements in the context."
                           "If the answer cannot be found in the context, say 'Not found.'"
            },

            # User prompt containing context + question
            {
                "role": "user",
                "content": f"""
                            Use the context below to answer the question.

                            Context:
                            {context}

                            Question:
                            {query}

                            Answer:
                            """
            }

        ],

        # Maximum number of tokens allowed in the answer
        max_tokens=500,
        # Temperature = 0 ensures deterministic responses
        temperature=0
    )

    # Extract model answer
    answer = response.choices[0].message.content

    return answer, sources

# main function
def main():

    # Folder containing promotion regulation documents
    folder_path = "C:/Users/amour/OneDrive - University of Sharjah/amira 3rd year/spring/junior project/promotion documents"

    # Step 1: Extract text from all PDFs
    documents = extract_text_from_pdfs(folder_path)

    # Step 2: Chunk documents
    all_chunks = []

    for doc in documents:

        # Clean document text
        cleaned = clean_text(doc["text"])
        # Create chunks
        doc_chunks = chunk_text(cleaned, doc["source"])
        # Add chunks to global list
        all_chunks.extend(doc_chunks)

    # Prepare data for BM25 keyword search
    tokenized_chunks = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # Step 3: Load embedding model
    model = SentenceTransformer("BAAI/bge-small-en")

    # Step 4: Build FAISS index
    index = build_index(all_chunks, model)

    # Step 5: Interactive chatbot loop
    while True:

        print("You: ", end="", flush=True)

        # Read user question
        query = input()

        # Retrieve relevant document chunks
        retrieved = retrieve(query, all_chunks, index, model, bm25)

        # Generate answer using LLM
        answer, sources = generate_answer(query, retrieved)

        # Print answer and sources
        print("\nBot:", answer)
        print("Sources:", ", ".join(sources))
        print("-" * 50)


# Run program
if __name__ == "__main__":
    main()
