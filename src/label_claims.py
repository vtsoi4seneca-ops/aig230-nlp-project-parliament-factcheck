#!/usr/bin/env python3
"""
Label extracted claims for fact-checking training.
Usage: uv run python src/label_claims.py [--input FILE] [--output FILE]
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Label claims for training')
    parser.add_argument('--input', default='data/extracted_claims.json', help='Input claims file')
    parser.add_argument('--output', default='data/labeled_claims.json', help='Output labeled file')
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run extract_claims.py first:")
        print("  uv run python src/extract_claims.py --limit 100")
        sys.exit(1)
    
    # Load existing claims
    with open(input_file) as f:
        claims = json.load(f)
    
    # Load existing labels if any (to resume)
    labeled_ids = set()
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
            labeled_ids = {c['context']['statement_id'] for c in existing}
        print(f"📋 Resuming: {len(labeled_ids)} already labeled")
    
    print(f"\n📝 Labeling {len(claims)} claims")
    print("Commands: t=TRUE, f=FALSE, m=MISLEADING, u=UNVERIFIED, s=SKIP, q=QUIT")
    print("-" * 80)
    
    labeled = []
    skipped = 0
    
    for i, claim in enumerate(claims, 1):
        # Skip already labeled
        if claim['context']['statement_id'] in labeled_ids:
            continue
        
        print(f"\n[{i}/{len(claims)}] {claim['speaker']}")
        print(f"Date: {claim['context']['date']}")
        print(f"Type: {claim['claim_type']}")
        print(f"\nClaim: {claim['claim_text']}")
        print("-" * 80)
        
        while True:
            cmd = input("Label (t/f/m/u/s/q): ").strip().lower()
            
            if cmd == 'q':
                print("\n💾 Saving progress...")
                break
            elif cmd == 's':
                skipped += 1
                print("  ⏭️  Skipped")
                break
            elif cmd in ['t', 'f', 'm', 'u']:
                label_map = {'t': 'TRUE', 'f': 'FALSE', 'm': 'MISLEADING', 'u': 'UNVERIFIED'}
                
                # Get explanation
                explanation = input("Explanation: ").strip()
                if not explanation:
                    explanation = "No explanation provided"
                
                # Get evidence source
                evidence = input("Evidence source (URL/doc): ").strip()
                if not evidence:
                    evidence = "Parliamentary Hansard records"
                
                claim['verification_status'] = label_map[cmd]
                claim['explanation'] = explanation
                claim['evidence_source'] = evidence
                claim['labeled_by'] = 'vinnietsoi'  # Change to your name
                claim['labeled_date'] = datetime.now().isoformat()
                
                labeled.append(claim)
                print(f"  ✅ {label_map[cmd]}")
                break
            else:
                print("  ❌ Invalid. Use: t/f/m/u/s/q")
        
        # Auto-save every 5 labels
        if len(labeled) % 5 == 0 and labeled:
            save_progress(output_file, labeled, claims, skipped)
    
    # Final save
    save_progress(output_file, labeled, claims, skipped)
    
    print(f"\n✅ Complete!")
    print(f"   Labeled: {len(labeled)}")
    print(f"   Skipped: {skipped}")
    print(f"   Total: {len(labeled) + len([c for c in claims if c['context']['statement_id'] in labeled_ids])}")
    print(f"\nNext step: Train model")
    print(f"   uv run python src/train.py")

def save_progress(output_file, new_labels, all_claims, skipped):
    """Merge new labels with existing and save"""
    existing = []
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    
    # Merge avoiding duplicates
    existing_ids = {c['context']['statement_id'] for c in existing}
    merged = existing + [c for c in new_labels if c['context']['statement_id'] not in existing_ids]
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"💾 Saved {len(merged)} total labels to {output_file}")

if __name__ == '__main__':
    main()
