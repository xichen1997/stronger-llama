import asyncio
import sys
from prompt_enhancer import PromptEnhancer

async def main():
    try:
        # Initialize the prompt enhancer with a specific model
        enhancer = PromptEnhancer(model_name="llama3.1:8b")

        # Example question
        question = '''
        I'm playing assetto corsa competizione, and I need you to tell me how many liters of fuel to take in a race. The qualifying time was 2:04.317, the race is 20 minutes long, and the car uses 2.73 liters per lap.'''
        print("Sending request to Ollama server...")
        # Get enhanced response with both chain-of-thought and reflection
        response = await enhancer.get_enhanced_response(
            question=question,
            use_cot=True,
            use_reflection=True
        )

        # Print the chain-of-thought response
        print("\n=== Chain of Thought Analysis ===")
        print(response['chain_of_thought'])

        # Print the reflection and improvements
        print("\n=== Reflection and Improvements ===")
        print(response['reflection'])

        # Evaluate response quality
        quality_metrics = enhancer.evaluate_response_quality(response)
        print("\n=== Quality Metrics ===")
        for metric, score in quality_metrics.items():
            print(f"{metric}: {score}")

        reasoning_queries = await enhancer.generate_reasoning_queries(question, response['chain_of_thought'], response['reflection'])
        print("\n=== Reasoning Queries ===")
        print(reasoning_queries)
        

    except ConnectionRefusedError:
        print("\nError: Could not connect to Ollama server!")
        print("\nPlease make sure to:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Start the Ollama server by running 'ollama serve' in a terminal")
        print("3. Pull the required model by running 'ollama pull llama3.2:3b'")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("Make sure Ollama server is running (run 'ollama serve' in a terminal)")
    # Run the async example
    asyncio.run(main()) 