import streamlit as st
from data_loader import HansardDB, DebateProcessor
import json

def labeling_interface():
    st.title("Canadian Parliament Claim Labeling Tool")
    
    db = HansardDB()
    processor = DebateProcessor()
    
    # Load unlabeled debates
    unlabeled = db.get_debates(limit=100)
    
    st.session_state.setdefault('current_idx', 0)
    idx = st.session_state['current_idx']
    
    if idx < len(unlabeled):
        row = unlabeled.iloc[idx]
        
        st.subheader(f"Debate: {row['debate_date']} - {row['speaker_name']}")
        st.write(row['content'])
        
        # Extract claims automatically
        claims = processor.extract_claims(
            row['content'], 
            row['speaker_name'],
            {'date': row['debate_date'], 'topic': row.get('topic')}
        )
        
        if claims:
            st.write("---")
            st.write("**Detected Claims:**")
            
            for i, claim in enumerate(claims):
                st.write(f"{i+1}. {claim['claim_text']}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button(f"True {i}", key=f"true_{i}"):
                        save_label(claim, "TRUE")
                with col2:
                    if st.button(f"False {i}", key=f"false_{i}"):
                        save_label(claim, "FALSE")
                with col3:
                    if st.button(f"Misleading {i}", key=f"misleading_{i}"):
                        save_label(claim, "MISLEADING")
                with col4:
                    if st.button(f"Skip {i}", key=f"skip_{i}"):
                        continue
        
        if st.button("Next Debate"):
            st.session_state['current_idx'] += 1
            st.rerun()

def save_label(claim: dict, verdict: str):
    """Save labeled claim to database"""
    # Implementation to save to your PostgreSQL fact_claims table
    pass

if __name__ == "__main__":
    labeling_interface()

