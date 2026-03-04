from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from typing import List, Dict, Tuple
import os
import re
import spacy

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
        """Retrieve debate statements from PostgreSQL with document info"""
        query = """
        SELECT 
            s.id,
            s.document_id,
            d.date as debate_date,
            d.session_id,
            d.number as document_number,
            s.time,
            s.who_en as speaker_name,
            s.who_context_en as speaker_context,
            s.member_id,
            s.politician_id,
            s.h1_en as topic_h1,
            s.h2_en as topic_h2,
            s.h3_en as topic_h3,
            s.content_en as content,
            s.content_fr,
            s.sequence,
            s.wordcount,
            s.procedural,
            s.statement_type,
            s.written_question,
            s.bill_debate_stage,
            s.bill_debated_id
        FROM hansards_statement s
        JOIN hansards_document d ON s.document_id = d.id
        WHERE 1=1
        AND s.content_en IS NOT NULL
        AND s.content_en != ''
        """
        
        params = {}
        if start_date:
            query += " AND d.date >= :start_date"
            params['start_date'] = start_date
        if end_date:
            query += " AND d.date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY d.date DESC, s.sequence"
        
        if limit:
            query += f" LIMIT {limit}"
            
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    
    def get_debate_stats(self) -> Dict:
        """Get statistics about the debate data"""
        stats = {}
        
        with self.engine.connect() as conn:
            # Total statements
            result = conn.execute(text("SELECT COUNT(*) FROM hansards_statement"))
            stats['total_statements'] = result.scalar()
            
            # Date range
            result = conn.execute(text("""
                SELECT MIN(date), MAX(date) 
                FROM hansards_document 
                WHERE public = true
            """))
            min_date, max_date = result.fetchone()
            stats['date_range'] = f"{min_date} to {max_date}"
            
            # Unique speakers
            result = conn.execute(text("""
                SELECT COUNT(DISTINCT who_en) 
                FROM hansards_statement 
                WHERE who_en IS NOT NULL AND who_en != ''
            """))
            stats['unique_speakers'] = result.scalar()
            
        return stats

    def get_verified_claims(self) -> pd.DataFrame:
        """Retrieve fact-checked claims for training labels"""
        # This table doesn't exist yet - we'll create it later
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
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn)
        except:
            # Return empty dataframe if table doesn't exist
            return pd.DataFrame(columns=[
                'claim_text', 'speaker_name', 'debate_date',
                'verification_status', 'evidence_source', 'explanation'
            ])


class DebateProcessor:
    def __init__(self):
        # Load spaCy for sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

    def extract_claims(self, text: str, speaker: str, context: Dict) -> List[Dict]:
        """
        Extract factual claims from debate text using linguistic patterns
        """
        # Clean HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        doc = self.nlp(clean_text)
        claims = []

        # Patterns indicating factual assertions
        fact_patterns = [
            r'\b(statistics? show|data indicates?|according to|research finds?)\b',
            r'\b(\d+\s*(percent|billion|million|thousand))\b',
            r'\b(unemployment|inflation|GDP|economy|growth|deficit|debt)\s+(is|at|reached?)\b',
            r'\b(legislation|bill|act)\s+(number|passed?|introduced)\b',
            r'\b(voted?|supported?|opposed?)\s+(for|against)\b'
        ]

        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Check if sentence contains factual patterns
            is_factual = any(re.search(pattern, sent_text, re.IGNORECASE)
                           for pattern in fact_patterns)

            if is_factual and len(sent_text) > 20:
                claims.append({
                    'claim_text': sent_text,
                    'full_context': clean_text,
                    'speaker': speaker,
                    'context': context,
                    'claim_type': self._classify_claim_type(sent_text)
                })

        return claims

    def _classify_claim_type(self, text: str) -> str:
        """Classify the type of factual claim"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['economy', 'gdp', 'inflation', 'unemployment', 'deficit']):
            return 'economic'
        elif any(word in text_lower for word in ['climate', 'environment', 'carbon', 'emission']):
            return 'environmental'
        elif any(word in text_lower for word in ['health', 'hospital', 'medicare', 'vaccine']):
            return 'health'
        elif any(word in text_lower for word in ['bill', 'legislation', 'act', 'law']):
            return 'legislative'
        else:
            return 'general'

    def create_training_pairs(self, claims_df: pd.DataFrame) -> List[Dict]:
        """
        Create input-output pairs for LLM fine-tuning
        Format: Claim + Context -> Verification Label + Explanation
        """
        training_data = []

        for _, row in claims_df.iterrows():
            # Input prompt
            prompt = f"""Fact-check the following claim made in Canadian Parliament:

Claim: "{row['claim_text']}"
Speaker: {row['speaker_name']} ({row.get('party', 'Unknown')})
Date: {row['debate_date']}
Topic: {row.get('topic', 'General Debate')}

Context: {row['full_context'][:500]}...

Is this claim accurate? Provide evidence and explanation."""

            # Expected output (for supervised fine-tuning)
            completion = f"""Verification: {row['verification_status']}

Evidence: {row.get('evidence_source', 'No source provided')}

Explanation: {row['explanation']}

Confidence: {row.get('confidence_score', 'Medium')}"""

            training_data.append({
                'prompt': prompt,
                'completion': completion,
                'metadata': {
                    'claim_type': row.get('claim_type', 'general'),
                    'parliament': row.get('parliament'),
                    'session': row.get('session')
                }
            })

        return training_data
