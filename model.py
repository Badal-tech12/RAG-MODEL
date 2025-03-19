import pandas as pd
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
df = pd.read_csv('metadata.csv', nrows=10000)
df = df.dropna(subset=['abstract'])
documents = []
for idx, row in df.iterrows():
    documents.append({
        "content": row['abstract'][:500],  # Truncate abstract to 500 characters
        "meta": {
            "title": row['title'],
            "authors": row['authors'],
            "publish_time": row['publish_time'],
            "source": row['source_x']
        }
    })
    document_store = FAISSDocumentStore(
    sql_url="sqlite:///faiss_document_store.db",
    faiss_index_factory_str="Flat",
    embedding_dim=384  # Set embedding dimension to 384
)

# Write documents to the document store
document_store.write_documents(documents)
# Step 6: Initialize the Retriever (Embedding-based)
# Use a lightweight embedding model for semantic search
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and effective model
)
document_store.update_embeddings(retriever)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)
question = "What is the impact of climate change on biodiversity?"
results = pipeline.run(query=question, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 3}})
print(f"Question: {question}")
for idx, answer in enumerate(results["answers"]):
    print(f"\nAnswer {idx + 1}:")
    print(f"  - Answer: {answer.answer}")
    print(f"  - Confidence: {answer.score:.4f}")
    print(f"  - Context: {answer.context}")
    print(f"  - Metadata: {answer.meta}")
    
