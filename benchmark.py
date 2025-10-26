"""
Egységes RAG vs Full Context AI Benchmark
Kombinált verzió - RAG rendszer + AI tesztelés egy fájlban
"""

import os
import time
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# AI API imports
import openai
import google.generativeai as genai

# RAG komponensek
import chromadb
from sentence_transformers import SentenceTransformer

# Környezeti változók betöltése
load_dotenv()

# ============================================================================
# RAG RENDSZER KOMPONENSEK
# ============================================================================

def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    """Szöveg darabolása"""
    # Szekciók szerint darabolás (## alapján)
    sections = re.split(r'\n## ', text)
    chunks = []
    
    for section in sections:
        if len(section.strip()) > 50:  # Túl rövid szekciók kihagyása
            # Ha a szekció túl hosszú, tovább darabolás
            if len(section) > chunk_size:
                sentences = re.split(r'[.!?]\n', section)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) < chunk_size:
                        current_chunk += sentence + "\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + "\n"
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(section.strip())
    
    return chunks

def setup_rag_system():
    """RAG rendszer beállítása"""
    print("RAG rendszer beállítása...")
    
    # Modell és ChromaDB
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.Client()
    
    # Collection törlése és újra létrehozása
    try:
        client.delete_collection("clearservice_docs")
    except:
        pass
    
    collection = client.create_collection("clearservice_docs")
    
    # Dokumentum betöltése
    with open('data/topics.txt', 'r', encoding='utf-8') as f:
        context = f.read()
    
    # Darabolás és embedding
    chunks = chunk_text(context)
    print(f"Dokumentum darabolva {len(chunks)} részre")
    
    embeddings = model.encode(chunks)
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[f"chunk_{i}"]
        )
    
    print("RAG rendszer beállítva!")
    return model, collection, context

def rag_retrieve(model, collection, question: str, top_k: int = 3):
    """RAG lekérdezés"""
    question_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k
    )
    return results['documents'][0], results['distances'][0]

# ============================================================================
# ÉRTÉKELÉSI FÜGGVÉNYEK
# ============================================================================

def evaluate_answer_quality(predicted: str, expected: str) -> dict:
    """Válasz minőség értékelése"""
    predicted_lower = predicted.lower()
    expected_lower = expected.lower()
    
    expected_words = set(expected_lower.split())
    predicted_words = set(predicted_lower.split())
    
    if len(expected_words) == 0:
        return {'exact_match': 0, 'word_overlap': 0, 'contains_key_info': 0}
    
    word_overlap = len(expected_words.intersection(predicted_words)) / len(expected_words)
    exact_match = 1 if expected_lower in predicted_lower else 0
    contains_key_info = 1 if any(word in predicted_lower for word in expected_words if len(word) > 3) else 0
    
    return {
        'exact_match': exact_match,
        'word_overlap': word_overlap,
        'contains_key_info': contains_key_info
    }

# ============================================================================
# AI BENCHMARK OSZTÁLY
# ============================================================================

class AIBenchmark:
    def __init__(self):
        """AI benchmark OpenAI + Gemini 2.0-flash"""
        
        print("🚀 AI Benchmark inicializálva:")
        
        # OpenAI konfiguráció
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            openai.api_key = openai_key
            print("✅ OpenAI API kulcs betöltve")
        else:
            print("❌ OpenAI API kulcs hiányzik!")
        
        # Gemini konfiguráció
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            genai.configure(api_key=gemini_key)
            print("✅ Gemini API kulcs betöltve")
        else:
            print("❌ Gemini API kulcs hiányzik!")
        
        # Rate limiting
        self.last_gemini_call = 0
        self.gemini_calls_this_minute = 0
        self.gemini_call_times = []
        
        print("🔧 Rate limiting beállítva: 8 kérés/perc Gemini Free Tier")
    
    def wait_for_gemini_rate_limit(self):
        """Gemini Free Tier rate limiting kezelése"""
        current_time = time.time()
        
        # Régi hívásiidők törlése (1 percnél régebbiek)
        self.gemini_call_times = [t for t in self.gemini_call_times if current_time - t < 60]
        
        # Ha túl sok hívás volt az elmúlt percben, várakozás
        if len(self.gemini_call_times) >= 8:
            wait_time = 60 - (current_time - self.gemini_call_times[0]) + 1
            if wait_time > 0:
                print(f"⏳ Rate limit - várakozás {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        # Jelenlegi hívás regisztrálása
        self.gemini_call_times.append(current_time)
    
    def get_openai_response(self, prompt: str) -> tuple:
        """OpenAI válasz generálás"""
        try:
            start_time = time.time()
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            generation_time = time.time() - start_time
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Költség becslés: $0.375/1M input token, $1.5/1M output token
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * 0.375 + output_tokens * 1.5) / 1_000_000
            
            return answer, generation_time, tokens_used, cost, None
            
        except Exception as e:
            return None, 0, 0, 0, str(e)
    
    def get_gemini_response(self, prompt: str) -> tuple:
        """Gemini válasz generálás optimalizált prompttal"""
        try:
            self.wait_for_gemini_rate_limit()
            start_time = time.time()
            
            # Optimalizált prompt Gemini-nek (biztonságos)
            safe_prompt = prompt.replace("Clearservice", "szervezet").replace("cég", "szervezet")
            
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            )
            
            response = model.generate_content(
                safe_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.1,
                )
            )
            
            generation_time = time.time() - start_time
            
            if response.candidates and response.candidates[0].content:
                answer = response.candidates[0].content.parts[0].text
                
                # Token becslés (körülbelüli)
                tokens_used = len(safe_prompt.split()) + len(answer.split())
                
                # Költség becslés: $0.075/1M token (2.0-flash)
                cost = tokens_used * 0.075 / 1_000_000
                
                return answer, generation_time, tokens_used, cost, None
            else:
                return None, generation_time, 0, 0, "Nincs válasz generálva"
                
        except Exception as e:
            return None, 0, 0, 0, str(e)
    
    def create_prompt(self, context: str, question: str, method: str) -> str:
        """Optimalizált prompt létrehozása"""
        if method == "RAG":
            base_prompt = f"""Felhasználói kérdés megválaszolása dokumentum alapján.

Dokumentum tartalma:
{context}

Felhasználó kérdése:
{question}

Kérlek adj egy rövid, faktaalapú választ a dokumentum információi alapján:"""
        else:  # FullContext
            base_prompt = f"""Felhasználói kérdés megválaszolása teljes dokumentum alapján.

Teljes dokumentum:
{context}

Felhasználó kérdése:
{question}

Kérlek adj egy rövid, faktaalapú választ a dokumentum információi alapján:"""
        
        return base_prompt

def run_benchmark():
    """Fő benchmark futtatás"""
    print("🚀 RAG vs Full Context AI Benchmark")
    print("=" * 60)
    
    # Teszt méret választása
    print("\\nVálassz teszt méretet:")
    print("1. Gyors teszt (3 kérdés)")
    print("2. Közepes teszt (10 kérdés)")
    print("3. TELJES TESZT (50 kérdés)")
    
    choice = input("\\nVálasztás (1-3): ").strip()
    
    if choice == "1":
        test_size = 3
    elif choice == "2":
        test_size = 10
    elif choice == "3":
        test_size = 50
        confirm = input("\\n⚠️  50 kérdés ~40 perc Gemini Free Tier-rel. Folytatod? (igen/nem): ").strip().lower()
        if confirm not in ['igen', 'yes', 'y']:
            print("Teszt megszakítva.")
            return
    else:
        print("Érvénytelen választás!")
        return
    
    print(f"\\n🎯 {test_size} kérdéses teszt indítása...")
    
    # RAG rendszer beállítása
    model, collection, full_context = setup_rag_system()
    
    # AI benchmark inicializálása
    ai_benchmark = AIBenchmark()
    
    # Kérdések betöltése
    df = pd.read_csv('data/cs_qa.csv', encoding='utf-8')
    df = df.head(test_size)
    
    # Benchmark futtatás
    results = []
    start_time = datetime.now()
    
    print(f"\\n📊 Benchmark futtatás - {len(df)} kérdés...")
    print("-" * 60)
    
    for idx, row in df.iterrows():
        question = row['question']
        expected_answer = row['answer']
        topic = row['topic']
        
        print(f"\\n🔍 Kérdés {idx+1}/{len(df)}: {question[:60]}...")
        
        # RAG módszer
        rag_docs, rag_distances = rag_retrieve(model, collection, question, top_k=3)
        rag_context = "\\n\\n".join(rag_docs)
        
        # Tesztelés mindkét módszerrel és AI modellel
        for method, context in [("RAG", rag_context), ("FullContext", full_context)]:
            for ai_model in ["OpenAI", "Gemini"]:
                prompt = ai_benchmark.create_prompt(context, question, method)
                
                if ai_model == "OpenAI":
                    answer, gen_time, tokens, cost, error = ai_benchmark.get_openai_response(prompt)
                else:
                    answer, gen_time, tokens, cost, error = ai_benchmark.get_gemini_response(prompt)
                
                if error is None and answer:
                    # Minőség értékelés
                    quality_metrics = evaluate_answer_quality(answer, expected_answer)
                    
                    result = {
                        'question_id': idx,
                        'question': question,
                        'expected_answer': expected_answer,
                        'topic': topic,
                        'method': method,
                        'ai_model': ai_model,
                        'context_size': len(context),
                        'generated_answer': answer,
                        'generation_time': gen_time,
                        'tokens_used': tokens,
                        'cost_estimate': cost,
                        'error': None,
                        **quality_metrics
                    }
                    
                    print(f"  ✅ {method} + {ai_model}: {quality_metrics['word_overlap']:.2f} overlap, {gen_time:.2f}s")
                else:
                    result = {
                        'question_id': idx,
                        'question': question,
                        'expected_answer': expected_answer,
                        'topic': topic,
                        'method': method,
                        'ai_model': ai_model,
                        'context_size': len(context),
                        'generated_answer': None,
                        'generation_time': gen_time,
                        'tokens_used': tokens,
                        'cost_estimate': cost,
                        'error': error,
                        'exact_match': 0,
                        'word_overlap': 0,
                        'contains_key_info': 0
                    }
                    
                    print(f"  ❌ {method} + {ai_model}: HIBA - {error}")
                
                results.append(result)
        
        # Progress tracking és köztes mentés
        if (idx + 1) % 5 == 0:
            print(f"\\n📊 Haladás: {idx+1}/{len(df)} kérdés kész")
            successful = len([r for r in results if r['error'] is None])
            total = len(results)
            print(f"   Sikeres: {successful}/{total} ({successful/total*100:.1f}%)")
            print()
            
            # Köztes mentés docs mappába
            temp_df = pd.DataFrame(results)
            docs_path = 'docs'
            if not os.path.exists(docs_path):
                os.makedirs(docs_path)
            temp_filename = f'docs/temp_results_{idx+1}.csv'
            temp_df.to_csv(temp_filename, index=False, encoding='utf-8')
            print(f"💾 Köztes mentés: {temp_filename}")
            print("-" * 50)
            print()
    
    # Végső mentés docs mappába
    results_df = pd.DataFrame(results)
    docs_path = 'docs'
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    final_filename = f'docs/benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
    results_df.to_csv(final_filename, index=False, encoding='utf-8')
    
    elapsed_time = datetime.now() - start_time
    print(f"\\n🎉 Benchmark befejezve!")
    print(f"⏱️  Futási idő: {elapsed_time}")
    print(f"💾 Eredmények mentve: {final_filename}")
    
    # Gyors összefoglaló
    successful_results = results_df[results_df['error'].isna()]
    if len(successful_results) > 0:
        print(f"\\n📊 GYORS ÖSSZEFOGLALÓ:")
        print(f"Sikeres tesztek: {len(successful_results)}/{len(results_df)} ({len(successful_results)/len(results_df)*100:.1f}%)")
        
        avg_scores = successful_results.groupby(['method', 'ai_model'])['word_overlap'].mean()
        print("\\n🏆 Átlagos word overlap pontszámok:")
        for (method, model), score in avg_scores.items():
            print(f"  {method} + {model}: {score:.3f}")
        
        total_costs = successful_results.groupby('ai_model')['cost_estimate'].sum()
        print("\\n💰 Becsült összes költségek:")
        for model, cost in total_costs.items():
            print(f"  {model}: ${cost:.6f}")
    
    print(f"\\n🎨 Futtatás: python visualize_simple.py")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()