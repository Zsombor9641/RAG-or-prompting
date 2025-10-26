"""
Egyszerűsített vizualizációk a RAG vs Full Context benchmark eredményeihez
Csak 7 ábra: 6 alapvető + 1 témakörönkénti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Magyar nyelv támogatás
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_latest_results():
    """Legújabb benchmark eredmények betöltése docs mappából"""
    # Keresés eredményekre docs mappában és gyökérben is
    result_files = []
    
    # Először docs mappában
    docs_path = Path('docs')
    if docs_path.exists():
        result_files.extend(docs_path.glob('benchmark_results_*.csv'))
        result_files.extend(docs_path.glob('temp_optimized_results_50.csv'))
        result_files.extend(docs_path.glob('optimized_results_*.csv'))
        result_files.extend(docs_path.glob('temp_optimized_results_*.csv'))
        result_files.extend(docs_path.glob('temp_results_*.csv'))
    
    # Ha nincs a docs-ban, gyökérben keresés
    if not result_files:
        result_files.extend(Path('.').glob('benchmark_results_*.csv'))
        result_files.extend(Path('.').glob('temp_optimized_results_50.csv'))
        result_files.extend(Path('.').glob('optimized_results_*.csv'))
        result_files.extend(Path('.').glob('*results*.csv'))
    
    if not result_files:
        raise FileNotFoundError("Nincs találat eredmény fájl!")
    
    # Legújabb fájl
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Benchmark eredmények betöltése: {latest_file}")
    
    return pd.read_csv(latest_file, encoding='utf-8')

def analyze_data(df):
    """Alapvető adatelemzés"""
    print("="*60)
    print("ADATELEMZÉS ÖSSZEFOGLALÓ")
    print("="*60)
    
    print(f"Összes rekord: {len(df)}")
    print(f"Kérdések száma: {df['question_id'].nunique()}")
    print(f"Módszerek: {df['method'].unique()}")
    print(f"AI modellek: {df['ai_model'].unique()}")
    
    # Hibák
    errors = df[df['error'].notna()]
    successful = df[df['error'].isna()]
    
    print(f"\\nSikeres tesztek: {len(successful)} ({len(successful)/len(df)*100:.1f}%)")
    print(f"Hibás tesztek: {len(errors)} ({len(errors)/len(df)*100:.1f}%)")
    
    if len(errors) > 0:
        print("\\nHibák megoszlása:")
        error_breakdown = errors.groupby(['method', 'ai_model']).size()
        print(error_breakdown)
    
    return successful, errors

def create_simplified_charts(successful_df, filename):
    """Egyszerűsített grafikonok - csak 7 ábra"""
    
    if len(successful_df) == 0:
        print("Nincs sikeres adat a vizualizációhoz!")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # ELSŐ ÁBRA: 6 alapvető grafikon
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1.1 Válasz minőség (Word Overlap)
    ax1 = axes[0, 0]
    quality_data = successful_df.groupby(['method', 'ai_model'])['word_overlap'].mean().unstack()
    quality_data.plot(kind='bar', ax=ax1, color=['#2E8B57', '#FF6B6B'])
    ax1.set_title('Átlagos Válasz Minőség (Word Overlap)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Word Overlap Score')
    ax1.set_xlabel('Módszer')
    ax1.legend(title='AI Modell')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 1.2 Válaszidő összehasonlítás
    ax2 = axes[0, 1]
    time_data = successful_df.groupby(['method', 'ai_model'])['generation_time'].mean().unstack()
    time_data.plot(kind='bar', ax=ax2, color=['#4CAF50', '#FF9800'])
    ax2.set_title('Átlagos Válaszidő (másodperc)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Idő (s)')
    ax2.set_xlabel('Módszer')
    ax2.legend(title='AI Modell')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 1.3 Költség összehasonlítás
    ax3 = axes[0, 2]
    cost_data = successful_df.groupby(['method', 'ai_model'])['cost_estimate'].sum().unstack()
    cost_data.plot(kind='bar', ax=ax3, color=['#9C27B0', '#FF5722'])
    ax3.set_title('Összes Becsült Költség ($)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Költség ($)')
    ax3.set_xlabel('Módszer')
    ax3.legend(title='AI Modell')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 1.4 Kontextus méret hatása
    ax4 = axes[1, 0]
    context_effect = successful_df.groupby('method')[['context_size', 'word_overlap']].mean()
    
    ax4_twin = ax4.twinx()
    bars = ax4.bar(context_effect.index, context_effect['context_size'], alpha=0.7, color='lightblue', label='Kontextus méret')
    line = ax4_twin.plot(context_effect.index, context_effect['word_overlap'], color='red', marker='o', linewidth=3, markersize=8, label='Minőség')
    
    ax4.set_title('Kontextus Méret vs Minőség', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Kontextus méret (karakterek)', color='blue')
    ax4_twin.set_ylabel('Word Overlap Score', color='red')
    
    # 1.5 Token használat
    ax5 = axes[1, 1]
    successful_df['config'] = successful_df['method'] + ' + ' + successful_df['ai_model']
    sns.boxplot(data=successful_df, x='config', y='tokens_used', ax=ax5)
    ax5.set_title('Token Használat Eloszlása', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Konfiguráció')
    ax5.set_ylabel('Token szám')
    ax5.tick_params(axis='x', rotation=45)
    
    # 1.6 Kombinált pontszám
    ax6 = axes[1, 2]
    summary_stats = successful_df.groupby(['method', 'ai_model']).agg({
        'word_overlap': 'mean',
        'generation_time': 'mean',
        'cost_estimate': 'sum'
    })
    
    # Normalizálás és súlyozás
    summary_stats['speed_norm'] = 1 - (summary_stats['generation_time'] / summary_stats['generation_time'].max())
    summary_stats['cost_norm'] = 1 - (summary_stats['cost_estimate'] / summary_stats['cost_estimate'].max())
    
    summary_stats['combined_score'] = (
        summary_stats['word_overlap'] * 0.6 +  # Minőség 60%
        summary_stats['speed_norm'] * 0.2 +     # Sebesség 20%
        summary_stats['cost_norm'] * 0.2        # Költséghatékonyság 20%
    )
    
    final_scores = summary_stats['combined_score'].unstack()
    final_scores.plot(kind='bar', ax=ax6, color=['#00BCD4', '#FFC107'])
    ax6.set_title('Végső Kombinált Pontszám\\n(Minőség 60% + Sebesség 20% + Költség 20%)', 
                  fontsize=12, fontweight='bold')
    ax6.set_ylabel('Kombinált Pontszám')
    ax6.set_xlabel('Módszer')
    ax6.legend(title='AI Modell')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # PNG fájlok mentése docs mappába
    docs_path = 'docs'
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    main_chart_filename = f'docs/main_comparison_{filename[:-4]}.png'
    plt.savefig(main_chart_filename, dpi=300, bbox_inches='tight')
    print(f"Fő összehasonlító grafikon mentve: {main_chart_filename}")
    plt.show()
    
    # MÁSODIK ÁBRA: Témakörönkénti teljesítmény
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Témakörönkénti átlagok
    topic_performance = successful_df.groupby(['topic', 'method'])['word_overlap'].mean().unstack()
    
    # Heatmap
    sns.heatmap(topic_performance, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0.5, ax=ax, cbar_kws={'label': 'Word Overlap Score'})
    ax.set_title('Témakörönkénti Teljesítmény', fontsize=14, fontweight='bold')
    ax.set_xlabel('Módszer')
    ax.set_ylabel('Témakör')
    
    plt.tight_layout()
    
    # PNG fájl mentése docs mappába
    docs_path = 'docs'
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    topic_chart_filename = f'docs/topic_performance_{filename[:-4]}.png'
    plt.savefig(topic_chart_filename, dpi=300, bbox_inches='tight')
    print(f"Témakörönkénti teljesítmény grafikon mentve: {topic_chart_filename}")
    plt.show()
    
    return summary_stats

def print_summary_stats(successful_df, summary_stats):
    """Összefoglaló statisztikák kiírása"""
    print("\\n" + "="*60)
    print("RÉSZLETES STATISZTIKAI ÖSSZEFOGLALÓ")
    print("="*60)
    
    # Alapvető metrikák
    for method in ['RAG', 'FullContext']:
        for model in ['Gemini', 'OpenAI']:
            subset = successful_df[(successful_df['method'] == method) & 
                                 (successful_df['ai_model'] == model)]
            if len(subset) > 0:
                avg_quality = subset['word_overlap'].mean()
                avg_time = subset['generation_time'].mean()
                total_cost = subset['cost_estimate'].sum()
                
                print(f"\\n{method} + {model}:")
                print(f"  Átlagos minőség: {avg_quality:.3f}")
                print(f"  Átlagos idő: {avg_time:.2f}s")
                print(f"  Összes költség: ${total_cost:.5f}")
    
    # Kombinált pontszámok
    print(f"\\n📊 VÉGSŐ KOMBINÁLT PONTSZÁMOK:")
    for idx, score in summary_stats['combined_score'].items():
        method, model = idx
        print(f"  {method} + {model}: {score:.3f}")

def main():
    """Fő program - egyszerűsített vizualizáció"""
    try:
        # Adatok betöltése
        df = load_latest_results()
        
        # Alapvető elemzés
        successful_df, errors = analyze_data(df)
        
        if len(successful_df) == 0:
            print("❌ Nincs sikeres adat a vizualizációhoz!")
            return
        
        # Egyszerűsített grafikonok (7 ábra)
        summary_stats = create_simplified_charts(successful_df, 'optimized_results.csv')
        
        # Statisztikai összefoglaló
        print_summary_stats(successful_df, summary_stats)
        
        print("\n🎉 Egyszerűsített vizualizáció kész!")
        print("📊 Készült 2 ábra a docs/ mappában:")
        print("   1. docs/main_comparison_optimized_results.png (6 alapvető grafikon)")
        print("   2. docs/topic_performance_optimized_results.png (témakörönkénti teljesítmény)")
        print("\n📁 Minden output (CSV + PNG) mostantól a docs/ mappában mentődik.")
        print("🔄 A következő futtatásnál a program automatikusan ott keresi őket.")
        
    except Exception as e:
        print(f"❌ Hiba a vizualizáció során: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()