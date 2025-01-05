# LLM Prompting Framework

A Python framework for enhancing smaller language models (like Llama 3.2B) with advanced prompting techniques including chain-of-thought reasoning, reflection mechanisms, and structured reasoning to avoid hallucination.

## Features

- **Chain of Thought (CoT)**: Guides the model through step-by-step reasoning
- **Reflection Mechanism**: Enables the model to review and improve its initial responses
- **Structured Reasoning**: Helps prevent hallucination by encouraging explicit reasoning and uncertainty acknowledgment
- **Benchmarking Tools**: Evaluate and compare different prompting strategies

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd llm_prompting_framework
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure you have Ollama installed and the Llama model downloaded:

```bash
ollama pull llama2:3b
```

## Usage

### Basic Usage

```python
import asyncio
from src.prompt_enhancer import PromptEnhancer

async def main():
    enhancer = PromptEnhancer(model_name="llama2:3b")
    response = await enhancer.get_enhanced_response(
        "Your question here",
        use_cot=True,
        use_reflection=True,
        use_reasoning=True
    )
    print(response)

asyncio.run(main())
```

### Running Benchmarks

```python
from benchmarks.prompt_benchmark import PromptBenchmark

async def run_benchmark():
    benchmark = PromptBenchmark(model_name="llama2:3b")
    results = await benchmark.run_benchmark()
    benchmark.plot_results(results)
    benchmark.save_results(results)

asyncio.run(run_benchmark())
```

## Prompting Strategies

1. **Chain of Thought**

   - Breaks down complex questions into steps
   - Shows intermediate reasoning
   - Improves answer accuracy

2. **Reflection Mechanism**

   - Reviews initial responses
   - Identifies potential weaknesses
   - Suggests improvements
   - Provides revised answers

3. **Structured Reasoning**
   - Explicitly states assumptions
   - Indicates confidence levels
   - Notes limitations and uncertainties
   - Prevents hallucination

## Benchmarking

The framework includes tools to benchmark different prompting strategies:

- Response time measurement
- Quality metrics evaluation
- Visualization tools
- Results export

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
