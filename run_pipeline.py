#!/usr/bin/env python3
import argparse
import json
from src.data_loader import HansardDB, DebateProcessor
from src.train import train_model
from src.api import start_api

def main():
    parser = argparse.ArgumentParser(description="Canadian Parliament Fact-Checking Pipeline")
    parser.add_argument('--step', choices=['extract', 'label', 'train', 'serve', 'all'], 
                       default='all', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    if args.step in ['extract', 'all']:
        print("🔍 Step 1: Extracting claims from PostgreSQL...")
        db = HansardDB()
        processor = DebateProcessor()
        
        debates = db.get_debates(start_date="2020-01-01")
        all_claims = []
        
        for _, row in debates.iterrows():
            claims = processor.extract_claims(
                row['content'], 
                row['speaker_name'],
                row.to_dict()
            )
            all_claims.extend(claims)
        
        # Save for labeling
        with open('data/extracted_claims.json', 'w') as f:
            json.dump(all_claims, f, indent=2)
        print(f"✅ Extracted {len(all_claims)} claims")
    
    if args.step in ['train', 'all']:
        print("🚀 Step 2: Fine-tuning LLM...")
        # Load labeled data
        with open('data/labeled_claims.json', 'r') as f:
            training_data = json.load(f)
        
        train_model(training_data)
        print("✅ Model training complete")
    
    if args.step in ['serve', 'all']:
        print("🌐 Step 3: Starting API server...")
        start_api()

if __name__ == "__main__":
    main()

