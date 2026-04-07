import json
import sys
from pathlib import Path
from llama_cpp import Llama

MODEL_PATH = "/home/vinnietsoi/parliament-factcheck/models/factcheck-llama31-8b-fine-tuned-Q4_K_M.gguf"

def main(input_file="send-to-vince.json", output_file="inference_results.json"):
    print("Loading model on DGX Spark (GB10/Blackwell)...")
    
    # DGX Spark optimized settings
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,          # Full GPU offload to unified memory
        n_ctx=8192,               # Long context for parliamentary text
        n_batch=512,              # Optimal for GB10
        use_mmap=False,           # Critical for ARM64 stability
        use_mlock=False,
        verbose=True
    )
    
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for idx, item in enumerate(data):
        claim_id = item.get('metadata', {}).get('claim_id', f'item_{idx}')
        print(f"[{idx+1}/{len(data)}] Processing: {claim_id}")
        
        prompt = item['prompt']
        
        # Generate completion
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.1,      # Low temp for factual accuracy
            stop=["</s>", "\n\nClaim:", "Analyze this claim"],
            echo=False
        )
        
        generated_text = output['choices'][0]['text'].strip()
        
        results.append({
            'claim_id': claim_id,
            'generated': generated_text,
            'ground_truth': item.get('completion'),
            'prompt_length': len(prompt),
            'metadata': item.get('metadata', {})
        })
        
        # Print sample of output for monitoring
        if idx < 3:
            print(f"  Sample output: {generated_text[:100]}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Processed {len(results)} claims.")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    input_f = sys.argv[1] if len(sys.argv) > 1 else "send-to-vince.json"
    output_f = sys.argv[2] if len(sys.argv) > 2 else "inference_results.json"
    main(input_f, output_f)
