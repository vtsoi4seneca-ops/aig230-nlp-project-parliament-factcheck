#!/usr/bin/env python3
"""
Extract factual claims from Canadian parliamentary debates.
Usage: uv run python src/extract_claims.py [--limit N] [--start-date YYYY-MM-DD]
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import HansardDB, DebateProcessor


def main():
    parser = argparse.ArgumentParser(
        description='Extract factual claims from parliamentary debates'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        default=100,
        help='Number of debate statements to process (default: 100)'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='2024-01-01',
        help='Start date for debates (default: 2024-01-01)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/extracted_claims.json',
        help='Output file path (default: data/extracted_claims.json)'
    )
    
    args = parser.parse_args()
    
    print('🔍 Step 4: Extracting factual claims from debates...')
    print(f'   Start date: {args.start_date}')
    print(f'   Limit: {args.limit} statements')
    print(f'   Output: {args.output}')
    
    # Initialize
    db = HansardDB()
    processor = DebateProcessor()
    
    # Get debates
    print(f'\n📥 Loading debates from database...')
    debates = db.get_debates(start_date=args.start_date, limit=args.limit)
    print(f'✅ Loaded {len(debates)} debate statements')
    
    # Extract claims
    print('\n🔎 Extracting factual claims...')
    all_claims = []
    
    for idx, row in debates.iterrows():
        # Skip procedural statements
        if row.get('procedural', False):
            continue
        
        # Skip empty speakers
        if not row.get('speaker_name'):
            continue
        
        context = {
            'date': str(row['debate_date']),
            'session': row['session_id'],
            'topic_h1': row.get('topic_h1', ''),
            'topic_h2': row.get('topic_h2', ''),
            'topic_h3': row.get('topic_h3', ''),
            'document_id': row['document_id'],
            'statement_id': row['id']
        }
        
        claims = processor.extract_claims(
            row['content'], 
            row['speaker_name'],
            context
        )
        
        all_claims.extend(claims)
        
        # Progress update every 10 rows
        if idx % 10 == 0 and idx > 0:
            print(f'  Processed {idx}/{len(debates)} statements, found {len(all_claims)} claims so far')
    
    print(f'\n✅ Extraction complete!')
    print(f'📊 Found {len(all_claims)} factual claims from {len(debates)} statements')
    print(f'   Extraction rate: {len(all_claims)/len(debates)*100:.1f}%')
    
    # Show sample claims
    if all_claims:
        print('\n📄 Sample claims:')
        for i, claim in enumerate(all_claims[:3], 1):
            print(f'\n{i}. [{claim["claim_type"].upper()}] {claim["speaker"]}')
            print(f'   Date: {claim["context"]["date"]}')
            print(f'   Topic: {claim["context"]["topic_h2"] or "N/A"}')
            print(f'   Claim: {claim["claim_text"][:150]}...')
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(all_claims, f, indent=2, default=str)
    
    print(f'\n💾 Saved {len(all_claims)} claims to {output_path}')
    print('\nNext steps:')
    print('  1. Label claims: uv run python src/labeling_tool.py')
    print('  2. Train model: uv run python src/train.py')
    print('  3. Run API: uv run python src/api.py')


if __name__ == '__main__':
    main()
