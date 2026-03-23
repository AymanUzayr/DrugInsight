import streamlit as st
import pandas as pd
import json

# Setup page style
st.set_page_config(page_title="DrugInsight", page_icon="💊", layout="centered")

# Custom CSS for modern, premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }
    h1 {
        background: linear-gradient(90deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        font-size: 3rem !important;
    }
    .subtitle {
        text-align: center;
        color: #8b949e;
        margin-bottom: 3rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .risk-major { color: #f85149; font-weight: bold; }
    .risk-moderate { color: #d29922; font-weight: bold; }
    .risk-minor { color: #2ea043; font-weight: bold; }
    
    /* Make selectboxes look nicer */
    .stSelectbox label {
        color: #e6edf3 !important;
        font-weight: 600;
    }
    div[data-baseweb="select"] {
        border-radius: 8px !important;
        border-color: #30363d;
    }
    
    /* Primary button restyling */
    button[kind="primary"] {
        background: linear-gradient(180deg, #238636, #2ea043) !important;
        border: 1px solid rgba(240, 246, 252, 0.1) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(180deg, #2ea043, #3fb950) !important;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    from drug_insight import DrugInsight
    return DrugInsight()


def main():
    st.markdown("<h1>DrugInsight</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Explainable Neural Drug-Drug Interaction Prediction</div>", unsafe_allow_html=True)

    with st.spinner("Initializing DrugInsight Framework..."):
        predictor = load_predictor()
        drug_list = predictor.drug_names()

    st.markdown("### Search Protocol")
    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.selectbox("First Compound", options=[""] + drug_list, index=0, help="Select primary substance")
    with col2:
        drug_b = st.selectbox("Second Compound", options=[""] + drug_list, index=0, help="Select secondary substance")

    st.write("") # Spacer
    if st.button("Synthesize Interaction Report", type="primary", use_container_width=True):
        if not drug_a or not drug_b:
            st.warning("⚠️ Please select both compounds to proceed with prediction.")
            return
            
        if drug_a == drug_b:
            st.error("⚠️ Selected compounds are identical. Please choose two distinct drugs.")
            return

        with st.spinner("Executing Graph Neural Network and Evidence Fusion..."):
            result = predictor.predict(drug_a, drug_b)
            st.session_state['result'] = result
            st.session_state['drug_a'] = drug_a
            st.session_state['drug_b'] = drug_b

    # Render results if available
    if 'result' in st.session_state:
        # Check if the inputs have changed since last prediction, clear if so
        if st.session_state.get('drug_a') != drug_a or st.session_state.get('drug_b') != drug_b:
            del st.session_state['result']
            st.rerun()

        result = st.session_state['result']
        if 'error' in result:
            st.error(f"Computation Error: {result['error']}")
            return

        st.divider()
        
        # Prediction Title
        interaction_found = result['interaction']
        sev_class = result['severity'].lower()
        
        if interaction_found:
            st.markdown(f"### ⚠️ Interaction Detected: <span class='risk-{sev_class}'>{result['severity']} Risk</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"### ✅ No Significant Interaction Expected", unsafe_allow_html=True)

        # Overview Metrics styling
        m1, m2, m3 = st.columns(3)
        m1.metric("Fusion Risk Index", f"{result['risk_index']} / 100")
        m2.metric("Ensemble Confidence", result['confidence'])
        m3.metric("Severity", result['severity'])

        st.info(f"**Clinical Summary:** {result['summary']}")

        # Tabs
        tab_mech, tab_ev, tab_scores = st.tabs(["Mechanism & Recommendation", "Data Sources", "Model Telemetry"])

        with tab_mech:
            st.markdown("#### Pharmacological Mechanism")
            st.markdown(f">{result['mechanism']}")
            st.markdown("#### Clinical Recommendation")
            st.markdown(f">**{result['recommendation']}**")
            
            with st.expander("View Full Explainer Output JSON"):
                st.write(result.get('full_explanation', 'No full explanation available.'))

        with tab_ev:
            ev = result['evidence']
            st.markdown("#### 🏥 DrugBank Curated Matrix")
            st.write(f"- **Shared Enzymes:** {ev['drugbank']['shared_enzymes'] or 'None'}")
            st.write(f"- **Shared Targets:** {ev['drugbank']['shared_targets'] or 'None'}")
            pathways = ev['drugbank'].get('shared_pathways', [])
            st.write(f"- **Shared Pathways:** {pathways if pathways else 'None'}")
            st.write(f"- **Known Interaction in Database:** {'Yes' if ev['drugbank']['known_interaction'] else 'No'}")

            st.write("")
            st.markdown("#### 📊 TWOSIDES Pharmacovigilance Data")
            prr = ev['twosides']['max_PRR']
            signal_text = f"PRR = {prr:.2f}" if prr > 0 else "No historical signal found"
            st.write(f"- **Signal Strength:** {signal_text}")
            if ev['twosides']['confounding_flag']:
                st.warning("⚠️ High PRR flagged: Potential confounding due to isolated drug toxicity.")

        with tab_scores:
            cs = result['component_scores']
            st.markdown("#### Sub-Model Score Breakdown")
            
            chart_data = pd.DataFrame({
                "Component": ["Rule-based (DrugBank)", "ML Model (GNN+MLP)", "TWOSIDES Enum"],
                "Score": [cs['rule_score'], cs['ml_score'], cs['twosides_score']],
                "Weight in Fusion": [cs['weights']['rule'], cs['weights']['ml'], cs['weights']['twosides']]
            })
            st.dataframe(chart_data, use_container_width=True, hide_index=True)
            
            unc = result['uncertainty']
            st.markdown("#### Source Confidences")
            st.code(
                f"DrugBank: {unc['drugbank_confidence']}\n"
                f"ML Model: {unc['ml_confidence']}\n"
                f"TWOSIDES: {unc['twosides_confidence']}\n"
                f"Overall:  {unc['overall_confidence']}",
                language="yaml"
            )

if __name__ == "__main__":
    main()
