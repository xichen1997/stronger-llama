import json
from typing import List, Dict, Any, Optional
import ollama
import httpx

class PromptEnhancer:
    def __init__(self, model_name: str = "llama3.2:3b"):
        """Initialize the prompt enhancer with specific model."""
        self.model_name = model_name
        self.client = ollama

    def generate_cot_prompt(self, question: str) -> str:
        """Generate a combined chain-of-thought and reasoning prompt."""
        return f"""Question: {question}
                Important: Base your response ONLY on verifiable information and logical reasoning. Let's solve this carefully, showing our reasoning for each step:

                1) First, let's identify:
                - What we know for certain
                - What assumptions we're making
                - What we need to determine
                - What logical steps connect our facts to conclusions

                2) Let's break this down systematically:
                - Key components of the problem
                - Relevant relationships or patterns
                - Potential approaches
                - Limitations of each approach

                3) For each step of our solution:
                - Explain the reasoning
                - Note confidence level (High/Medium/Low)
                - Highlight any uncertainties
                - Provide evidence or logical justification

                4) Finally, let's:
                - Combine our findings
                - Verify our logic
                - Address any uncertainties
                - State our conclusion with appropriate confidence

                If you're unsure about something, explicitly state that. Please show your complete reasoning process and clearly mark any assumptions or uncertainties."""

    def generate_reflection_prompt(self, initial_response: str) -> str:
        """Generate a reflection prompt to review and improve the answer."""
        return f"""Given this initial response:
                {initial_response}

                Let's reflect on this answer:
                1) What assumptions did we make?
                2) What could be potential weaknesses in our reasoning?
                3) Are there alternative perspectives we haven't considered?
                4) How confident are we in each part of our answer?

                Based on this reflection, please provide:
                1) A critique of the initial response
                2) Suggested improvements
                3) A revised answer if necessary"""

    async def get_enhanced_response(self, question: str, 
                                  use_cot: bool = True,
                                  use_reflection: bool = True) -> Dict[str, Any]:
        """Get an enhanced response using multiple prompting techniques."""
        responses = {}
        
        try:
            # Combined chain-of-thought and reasoning response
            if use_cot:
                cot_prompt = self.generate_cot_prompt(question)
                cot_response = self.client.generate(model=self.model_name, 
                                                  prompt=cot_prompt)
                responses['chain_of_thought'] = cot_response['response']

            # Reflection on previous responses
            if use_reflection and use_cot:
                initial_response = responses['chain_of_thought']
                reflection_prompt = self.generate_reflection_prompt(initial_response)
                reflection_response = self.client.generate(model=self.model_name,
                                                         prompt=reflection_prompt)
                responses['reflection'] = reflection_response['response']

            return responses
            
        except httpx.ConnectError:
            raise ConnectionRefusedError("Could not connect to Ollama server")
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    async def generate_reasoning_queries(self, question: str, cot_response: str, reflection_response: str) -> List[str]:
        """based on the question, cot response and reflection, generate a list of reasoning queries"""
        reasoning_queries = f"""based on the question: {question}, cot response and reflection, generate a short answer.
        
        COT output:
        {cot_response}
        
        Reflection output:
        {reflection_response}
        """
        reasoning_queries_response = self.client.generate(model=self.model_name, prompt=reasoning_queries)
        reasoning_queries = reasoning_queries_response['response']
        print(reasoning_queries)
        return reasoning_queries

    def evaluate_response_quality(self, response: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of the response based on various metrics."""
        # This is a placeholder for more sophisticated evaluation
        metrics = {
            'coherence': 0.0,
            'reasoning_depth': 0.0,
            'self_consistency': 0.0
        }
        
        # Add evaluation logic here
        # For now, returning placeholder metrics
        return metrics 