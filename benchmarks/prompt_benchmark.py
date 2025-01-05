import json
import time
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..src.prompt_enhancer import PromptEnhancer

class PromptBenchmark:
    def __init__(self, model_name: str = "llama2:3b"):
        self.enhancer = PromptEnhancer(model_name)
        self.test_questions = [
            "Explain how photosynthesis works",
            "What causes climate change?",
            "How does a computer's CPU work?",
            "Explain the theory of relativity",
            "What is the difference between RNA and DNA?"
        ]
        
    async def run_benchmark(self, 
                          combinations: List[Dict[str, bool]] = None) -> pd.DataFrame:
        """Run benchmark with different prompting combinations."""
        if combinations is None:
            combinations = [
                {'use_cot': True, 'use_reflection': False, 'use_reasoning': False},
                {'use_cot': True, 'use_reflection': True, 'use_reasoning': False},
                {'use_cot': True, 'use_reflection': True, 'use_reasoning': True},
            ]
        
        results = []
        for question in tqdm(self.test_questions, desc="Processing questions"):
            for combo in combinations:
                start_time = time.time()
                response = await self.enhancer.get_enhanced_response(
                    question,
                    **combo
                )
                end_time = time.time()
                
                metrics = self.enhancer.evaluate_response_quality(response)
                
                result = {
                    'question': question,
                    'strategy': str(combo),
                    'response_time': end_time - start_time,
                    **metrics,
                    'response': response
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_results(self, results: pd.DataFrame) -> None:
        """Plot benchmark results."""
        plt.figure(figsize=(12, 6))
        
        # Plot response times
        plt.subplot(1, 2, 1)
        results.boxplot(column=['response_time'], by='strategy')
        plt.title('Response Times by Strategy')
        plt.xticks(rotation=45)
        
        # Plot quality metrics
        plt.subplot(1, 2, 2)
        metrics = ['coherence', 'reasoning_depth', 'self_consistency']
        results[metrics + ['strategy']].groupby('strategy').mean().plot(kind='bar')
        plt.title('Quality Metrics by Strategy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        
    def save_results(self, results: pd.DataFrame, filename: str = 'benchmark_results.json'):
        """Save benchmark results to file."""
        results.to_json(filename, orient='records', indent=2) 