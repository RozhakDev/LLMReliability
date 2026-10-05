import streamlit as st
from typing import Dict, Any

def render_results(parsed_json: Dict[str, Any], pos: str, raw_text: str = "") -> None:
    """Render the evaluation results cleanly."""
    if not parsed_json:
        st.error("JSON Parsing Failed. Model violated schema constraints.")
        with st.expander("Show Raw Output"):
            st.code(raw_text)
        return
        
    winner = parsed_json.get("decision", "Unknown")
    
    if pos == "BA":
        if winner == "A":
            logical_winner = "Candidate 2 (Position A)"
            color = "#d2a8ff"
        elif winner == "B":
            logical_winner = "Candidate 1 (Position B)"
            color = "#58a6ff"
        else:
            logical_winner = "INVALID"
            color = "#ff7b72"
    else:
        if winner == "A":
            logical_winner = "Candidate 1 (A)"
            color = "#58a6ff"
        elif winner == "B":
            logical_winner = "Candidate 2 (B)"
            color = "#d2a8ff"
        else:
            logical_winner = "INVALID"
            color = "#ff7b72"
    
    st.markdown(f"<h3 style='text-align: left; color: {color}; margin-bottom: 24px;'>Decision: {logical_winner}</h3>", unsafe_allow_html=True)
    
    c_acc = parsed_json.get("criteria", {}).get("accuracy", {})
    c_rel = parsed_json.get("criteria", {}).get("relevance", {})
    c_clar = parsed_json.get("criteria", {}).get("clarity", {})
    
    c1, c2, c3 = st.columns(3)
    for col, title, data in zip([c1, c2, c3], ["Accuracy", "Relevance", "Clarity"], [c_acc, c_rel, c_clar]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{title}</div>
                <div class="metric-value">A: {data.get('A','-')} <span style='font-weight: 300; font-size: 1.5rem; color:#484f58;'>|</span> B: {data.get('B','-')}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    exp = parsed_json.get("explanation", {})
    
    st.markdown("#### Detailed Explanations")
    t1, t2, t3 = st.tabs(["Accuracy", "Relevance", "Clarity"])
    with t1:
        st.write(exp.get("accuracy", "No explanation provided."))
    with t2:
        st.write(exp.get("relevance", "No explanation provided."))
    with t3:
        st.write(exp.get("clarity", "No explanation provided."))
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Traceable Evidence")
    evidences = parsed_json.get("evidence", [])
    if evidences:
        for ev in evidences:
            criterion = str(ev.get('criterion', '')).upper()
            st.markdown(f"""
            <div class="evidence-block">
                <div class="evidence-meta">
                    <strong>Criterion:</strong> {criterion} | 
                    <strong>Candidate:</strong> {ev.get('candidate')} | 
                    <strong>Assessment:</strong> {ev.get('assessment')}
                </div>
                <em>"{ev.get('evidence')}"</em>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No evidence extracted.")
