import os   #for path
import pdfplumber   #for pdf text extraction
import faiss  #vertor database
import tiktoken #for tokens
from sentence_transformers import SentenceTransformer #to vectorise
from openai import OpenAI #llm model

client = OpenAI()

#extracting from pdf
def extract_text_from_pdfs(folder_path):
    documents = []   #documents array to collect them

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):    #all pdf files in the directory
            filepath = os.path.join(folder_path, filename)
            print(f"Reading {filename}...")

            with pdfplumber.open(filepath) as pdf:   #extracting text from pdf
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

            documents.append({    #adding the file name and text to the array
                "source": filename,
                "text": full_text
            })

    return documents

#cleaning text
def clean_text(text):
    text = text.replace("\n", " ")  #replacing new lines with spaces
    text = " ".join(text.split())   #remove extra spaces and lines then join with one space
    return text

#chunking
def chunk_text(text, source, chunk_size=400, overlap=100):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text) #encode texts to tokens

    chunks = []
    start = 0

    # loop to create the chunks with the set size and overlap
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_string = enc.decode(chunk_tokens)

        chunks.append({
            "text": chunk_string,
            "source": source
        })

        start += chunk_size - overlap

    return chunks

#building faiss using cosine similarity
def build_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]  #using the created cunks

    embeddings = model.encode(texts).astype("float32")  #create vectors
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index

#retriving the best 3 similar vectors
def retrieve(query, chunks, index, model, k=3):
    q_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)

    D, I = index.search(q_vec, k) #returning the similarity score and index

    retrieved_chunks = [chunks[i] for i in I[0]]
    return retrieved_chunks

#answer generation using llm, generating the answer and the source
def generate_answer(query, retrieved_chunks):
    context = ""
    sources = set()

    for chunk in retrieved_chunks:
        context += chunk["text"] + "\n\n"
        sources.add(chunk["source"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[ { #the answer is generated only using the context to avoid hallucinations
                "role": "system",
                "content": "Answer ONLY using the provided context. If the answer is not found, say 'Not found.'"
            },
            {
                #generating the answer based on the context and query
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        max_tokens=300,
        temperature=0
    )

    answer = response.choices[0].message.content
    return answer, sources

#main function
def main():
    folder_path = "C:/Users/amour/OneDrive - University of Sharjah/amira 3rd year/spring/junior project/promotion documents"

    #Extracting PDFs
    documents = extract_text_from_pdfs(folder_path)

    #Chunking documents
    all_chunks = []

    #cleaning the documents
    for doc in documents:
        cleaned = clean_text(doc["text"])
        doc_chunks = chunk_text(cleaned, doc["source"])
        all_chunks.extend(doc_chunks)

    #Loading embedding model
    model = SentenceTransformer("BAAI/bge-small-en")

    #Building FAISS index
    index = build_index(all_chunks, model)

    while True:
        print("You: ", end="", flush=True)
        query = input()

        retrieved = retrieve(query, all_chunks, index, model)
        answer, sources = generate_answer(query, retrieved)

        print("\nBot:", answer)
        print("Sources:", ", ".join(sources))
        print("-" * 50)


if __name__ == "__main__":
    main()