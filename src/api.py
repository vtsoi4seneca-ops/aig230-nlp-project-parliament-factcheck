from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re

app = FastAPI(title="Canadian Parliament Fact-Checker API")

class ClaimRequest(BaseModel):
    claim_text: str
    speaker_name: Optional[str] = None
    debate_date: Optional[str] = None
    context: Optional[str] = None

class FactCheckResponse(BaseModel):
    claim: str
    verdict: str  # "TRUE", "FALSE", "MISLEADING", "UNVERIFIED", "NEEDS_CONTEXT"
    confidence: float
    explanation: str
    sources: List[Dict]
    correction: Optional[str] = None

# Global model variables
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """Load fine-tuned model on startup"""
    global model, tokenizer
    
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    adapter_path = "./models/factcheck-llama-canadian/final"
    
    # Load with quantization for inference efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print("Model loaded successfully")

@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check(request: ClaimRequest):
    """Main fact-checking endpoint"""
    
    # Construct prompt
    prompt = f"""Analyze this claim from Canadian Parliament and provide a fact-check:

Claim: "{request.claim_text}"
{f"Speaker: {request.speaker_name}" if request.speaker_name else ""}
{f"Date: {request.debate_date}" if request.debate_date else ""}
{f"Context: {request.context[:300]}..." if request.context else ""}

Provide your assessment in this exact format:
VERDICT: [TRUE/FALSE/MISLEADING/UNVERIFIED/NEEDS_CONTEXT]
CONFIDENCE: [0-100]
EXPLANATION: [Detailed explanation with evidence]
CORRECTION: [If false or misleading, provide the correct information]"""

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,  # Low temperature for factual consistency
        do_sample=True,
        top_p=0.9,
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse structured output
    parsed = parse_factcheck_output(response_text)
    
    return FactCheckResponse(
        claim=request.claim_text,
        verdict=parsed.get('verdict', 'UNVERIFIED'),
        confidence=float(parsed.get('confidence', 50)) / 100,
        explanation=parsed.get('explanation', 'No explanation generated'),
        sources=[],  # Populate from evidence retriever
        correction=parsed.get('correction')
    )

def parse_factcheck_output(text: str) -> Dict:
    """Parse structured output from model"""
    result = {}
    
    verdict_match = re.search(r'VERDICT:\s*(TRUE|FALSE|MISLEADING|UNVERIFIED|NEEDS_CONTEXT)', text, re.IGNORECASE)
    if verdict_match:
        result['verdict'] = verdict_match.group(1).upper()
    
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
    if confidence_match:
        result['confidence'] = confidence_match.group(1)
    
    explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=CORRECTION:|$)', text, re.DOTALL)
    if explanation_match:
        result['explanation'] = explanation_match.group(1).strip()
    
    correction_match = re.search(r'CORRECTION:\s*(.+?)(?=\n|$)', text, re.DOTALL)
    if correction_match:
        result['correction'] = correction_match.group(1).strip()
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


