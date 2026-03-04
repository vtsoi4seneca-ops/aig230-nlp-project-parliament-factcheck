from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from typing import List, Dict
import os

class HansardDB:
    def __init__(self):
        self.connection_string = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'spark_user')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'secretpassword')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'spark_db')}"
        )
        self.engine = create_engine(self.connection_string)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_debates(self, start_date: str = None, end_date: str = None, 
                   limit: int = None) -> pd.DataFrame:
        """Retrieve debate records from PostgreSQL"""
        query = """
        SELECT 
            id, volume, number, session, parliament, 
            debate_date, speaker_name, affiliation, 
            intervention_type, content, topic
        FROM hansard_debates
        WHERE 1=1
        """
        
        params = {}
        if start_date:
            query += " AND debate_date >= :start_date"
            params['start_date'] = start_date
        if end_date:
            query += " AND debate_date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY debate_date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
            
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    
    def get_verified_claims(self) -> pd.DataFrame:
        """Retrieve fact-checked claims for training labels"""
        query = """
        SELECT 
            c.claim_text, 
            c.speaker_name, 
            c.debate_date,
            c.verification_status,
            c.evidence_source,
            c.explanation
        FROM fact_claims c
        WHERE c.verification_status IS NOT NULL
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)

# Usage example
if __name__ == "__main__":
    db = HansardDB()
    recent_debates = db.get_debates(start_date="2024-01-01", limit=1000)
    print(f"Loaded {len(recent_debates)} debate records")

