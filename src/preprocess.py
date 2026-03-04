#!/usr/bin/env python3
"""
Preprocess extracted claims for training.
Cleans HTML, normalizes text, filters low-quality data.
Usage: uv run python src/preprocess.py [--input FILE] [--output FILE]
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict


def clean_html(text: str) -> str:
    """Remove HTML tags and entities"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove HTML entities
    text = re.sub(r'&\w+;', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize text for training"""
    # Fix encoding issues
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    # Remove non-breaking spaces
    text = text.replace('\xa0', ' ')
    return text


def is_quality_claim(claim: Dict) -> bool:
    """Filter out low-quality claims"""
    text = claim.get('claim_text', '')
    
    # Too short
    if len(text) < 30:
        return False
    
    # Too long (likely not a single claim)
    if len(text) > 500:
        return False
    
    # Contains mostly numbers/symbols
    if len(re.findall(r'[a-zA-Z]', text)) < len(text) * 0.5:
        return False
    
    # Check for procedural text patterns to exclude
    procedural_patterns = [
        r'^\s*\(?\s*(The House|Mr\. Speaker|Madam Speaker|The Chair)',
        r'^\s*\(?\s*(In my opinion|I declare)',
        r'^\s*\(?\s*(It being|There being)',
        r'^\s*\(?\s*(Pursuant to|In accordance with)',
        r'^\s*\(?\s*(Some hon\. members|All those in favour)',
    ]
    
    for pattern in procedural_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True


def preprocess_claim(claim: Dict) -> Dict:
    """Clean and normalize a single claim"""
    # Clean HTML from all text fields
    claim['claim_text'] = clean_html(claim['claim_text'])
    claim['full_context'] = clean_html(claim['full_context'])
    
    # Normalize
    claim['claim_text'] = normalize_text(claim['claim_text'])
    claim['full_context'] = normalize_text(claim['full_context'])
    
    # Clean speaker name
    if claim.get('speaker'):
        claim['speaker'] = clean_html(claim['speaker']).strip()
    
    return claim


def main():
    parser = argparse.ArgumentParser(description='Preprocess claims')
    parser.add_argument('--input', default='data/extracted_claims.json')
    parser.add_argument('--output', default='data/processed_claims.json')
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"❌ Input not found: {input_file}")
        print("Run: uv run python src/extract_claims.py")
        return
    
    # Load raw claims
    with open(input_file) as f:
        claims = json.load(f)
    
    print(f"📥 Loaded {len(claims)} raw claims")
    
    # Preprocess each claim
    processed = []
    for claim in claims:
        try:
            clean_claim = preprocess_claim(claim)
            
            if is_quality_claim(clean_claim):
                processed.append(clean_claim)
            else:
                print(f"  ⚠️  Filtered low-quality claim: {clean_claim['claim_text'][:50]}...")
                
        except Exception as e:
            print(f"  ❌ Error processing claim: {e}")
            continue
    
    # Save processed claims
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2)
    
    print(f"\n✅ Preprocessing complete")
    print(f"   Input: {len(claims)} claims")
    print(f"   Output: {len(processed)} quality claims")
    print(f"   Removed: {len(claims) - len(processed)} low-quality")
    print(f"\n💾 Saved to {output_file}")
    print(f"\nNext step: Label the processed claims")
    print(f"   uv run python src/label_claims.py --input {output_file}")


if __name__ == '__main__':
    main()
