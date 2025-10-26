"""
RAG Rendszer implementáció ChromaDB + paraphrase-multilingual-MiniLM-L12-v2 használatával
"""

import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Tuple
import re

class RAGSystem:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """RAG rendszer inicializálása"""
        print(f"RAG rendszer inicializálása {model_name} modellel...")
        
        # Embedding modell betöltése
        self.embedding_model = SentenceTransformer(model_name)
        
        # ChromaDB kliens
        self.chroma_client = chromadb.Client()
        
        # Kollekció törölése ha létezik
        try:
            self.chroma_client.delete_collection("clearservice_docs")
        except:
            pass
            
        # Új kollekció létrehozása
        self.collection = self.chroma_client.create_collection(
            name="clearservice_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("RAG rendszer sikeresen inicializálva!")
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Szöveg darabolása átfedő chunkokra"""
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def load_context_to_vectordb(self, context_file: str):
        """Kontextus betöltése a vektoradatbázisba"""
        print("Kontextus betöltése a vektoradatbázisba...")
        
        with open(context_file, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # Szöveg darabolása
        chunks = self.chunk_text(full_text)
        print(f"Szöveg {len(chunks)} chunkra darabolva")
        
        # Embeddings generálása
        print("Embeddings generálása...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        
        # ChromaDB-be töltés
        print("Adatok betöltése ChromaDB-be...")
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids
        )
        
        print(f"Sikeresen betöltve {len(chunks)} chunk a vektoradatbázisba!")
        return len(chunks)
    
    def retrieve_relevant_chunks(self, question: str, top_k: int = 3) -> List[Dict]:
        """Releváns chunkök lekérése a kérdéshez"""
        # Kérdés embedding
        question_embedding = self.embedding_model.encode([question])
        
        # Keresés
        results = self.collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=top_k
        )
        
        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            retrieved_chunks.append({
                'text': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],  # cosine similarity
                'id': results['ids'][0][i]
            })
        
        return retrieved_chunks

class FullContextSystem:
    """Teljes kontextus használata minden kérdéshez"""
    
    def __init__(self, context_file: str):
        with open(context_file, 'r', encoding='utf-8') as f:
            self.full_context = f.read()
        print(f"Teljes kontextus betöltve: {len(self.full_context)} karakter")
    
    def get_context(self, question: str) -> str:
        """Teljes kontextus visszaadása minden kérdéshez"""
        return self.full_context


def measure_retrieval_quality(rag_system: RAGSystem, df: pd.DataFrame, top_k_values: List[int] = [1, 3, 5]):
    """RAG rendszer retrieval minőségének mérése"""
    
    print("\\nRetrieval minőség mérése...")
    results = []
    
    for idx, row in df.iterrows():
        question = row['question']
        correct_topic = row['topic']
        correct_answer = row['answer']
        
        # Különböző top_k értékekre
        for top_k in top_k_values:
            start_time = time.time()
            retrieved_chunks = rag_system.retrieve_relevant_chunks(question, top_k)
            retrieval_time = time.time() - start_time
            
            # Ellenőrizzük, hogy a helyes topic megtalálható-e a retrieved chunksban
            topic_found = any(correct_topic.lower() in chunk['text'].lower() 
                            for chunk in retrieved_chunks)
            
            # Ellenőrizzük, hogy a helyes válasz vagy annak része megtalálható-e
            answer_found = any(correct_answer.lower() in chunk['text'].lower() 
                             for chunk in retrieved_chunks)
            
            results.append({
                'question_id': idx,
                'question': question,
                'correct_topic': correct_topic,
                'correct_answer': correct_answer,
                'top_k': top_k,
                'retrieval_time': retrieval_time,
                'topic_found': topic_found,
                'answer_found': answer_found,
                'retrieved_chunks': len(retrieved_chunks),
                'best_score': retrieved_chunks[0]['score'] if retrieved_chunks else 0
            })
    
    return pd.DataFrame(results)


def calculate_metrics(results_df: pd.DataFrame):
    """Recall@k és MRR metrikák kiszámítása"""
    metrics = {}
    
    for top_k in results_df['top_k'].unique():
        k_results = results_df[results_df['top_k'] == top_k]
        
        # Recall@k (topic alapján)
        topic_recall = k_results['topic_found'].mean()
        
        # Recall@k (answer alapján)  
        answer_recall = k_results['answer_found'].mean()
        
        # Átlagos retrieval idő
        avg_time = k_results['retrieval_time'].mean()
        
        # Átlagos score
        avg_score = k_results['best_score'].mean()
        
        metrics[f'recall_topic@{top_k}'] = topic_recall
        metrics[f'recall_answer@{top_k}'] = answer_recall
        metrics[f'avg_time@{top_k}'] = avg_time
        metrics[f'avg_score@{top_k}'] = avg_score
    
    # MRR számítás (Mean Reciprocal Rank)
    # Itt egyszerűsítve: ha megtalálja a top_k-ban, akkor 1/k reciprocal rank
    mrr_scores = []
    for question_id in results_df['question_id'].unique():
        question_results = results_df[results_df['question_id'] == question_id].sort_values('top_k')
        
        reciprocal_rank = 0
        for _, row in question_results.iterrows():
            if row['answer_found']:
                reciprocal_rank = 1 / row['top_k']
                break
        
        mrr_scores.append(reciprocal_rank)
    
    metrics['mrr'] = np.mean(mrr_scores)
    
    return metrics


if __name__ == "__main__":
    # Teszt futtatás
    print("RAG rendszer tesztelése...")
    
    # Adatok betöltése
    df = pd.read_csv('data/cs_qa.csv', encoding='utf-8')
    
    # RAG rendszer inicializálása
    rag = RAGSystem()
    
    # Kontextus betöltése
    rag.load_context_to_vectordb('data/topics.txt')
    
    # Teszt lekérdezés
    test_question = "Mennyi a fizetés?"
    chunks = rag.retrieve_relevant_chunks(test_question, top_k=3)
    
    print(f"\\nTeszt kérdés: {test_question}")
    print("Talált chunkök:")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. Score: {chunk['score']:.3f}")
        print(f"   Text: {chunk['text'][:100]}...")
        print()