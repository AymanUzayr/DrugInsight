import os
import sys

import streamlit as st


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import DDIPredictor  # noqa: E402


st.set_page_config(
    page_title="DrugInsight",
    page_icon="🧬",
    layout="centered",
)


@st.cache_resource(show_spinner=True)
def load_predictor():
    return DDIPredictor(model_path=os.path.join(ROOT_DIR, "models", "ddi_model.pt"),
                        data_dir=os.path.join(ROOT_DIR, "data", "processed"))


def render_result(result: dict):
    st.subheader("Result")

    cols = st.columns(3)
    cols[0].metric("Interaction", "YES" if result["interaction"] else "NO")
    cols[1].metric("Severity", result["severity"])
    cols[2].metric("Risk index", f'{result["risk_index"]} / 100')

    st.write(f'**Probability (fused)**: `{result["probability"]}`')
    st.write(f'**Model confidence**: `{result["uncertainty"]["overall_confidence"]}`')

    st.subheader("Summary")
    st.write(result.get("summary", ""))

    with st.expander("Mechanism"):
        st.write(result.get("mechanism", ""))

    with st.expander("Recommendation"):
        st.write(result.get("recommendation", ""))

    with st.expander("Evidence + component scores"):
        st.json(
            {
                "drugs": {
                    "drug_a": result.get("drug_a"),
                    "drug_b": result.get("drug_b"),
                    "drugbank_id_a": result.get("drugbank_id_a"),
                    "drugbank_id_b": result.get("drugbank_id_b"),
                },
                "evidence": result.get("evidence", {}),
                "component_scores": result.get("component_scores", {}),
                "uncertainty": result.get("uncertainty", {}),
            }
        )

    with st.expander("Full explanation"):
        st.write(result.get("full_explanation", ""))


def main():
    st.title("DrugInsight")
    st.caption("Drug–drug interaction (DDI) risk prediction using curated evidence + a GNN model.")

    examples = [
        ("Warfarin", "Aspirin"),
        ("Warfarin", "Fluconazole"),
        ("Atorvastatin", "Clarithromycin"),
        ("Metformin", "Ibuprofen"),
    ]

    with st.sidebar:
        st.header("Quick start")
        picked = st.selectbox(
            "Example pairs",
            options=list(range(len(examples))),
            format_func=lambda i: f"{examples[i][0]} × {examples[i][1]}",
        )
        if st.button("Use example"):
            st.session_state["drug_a"] = examples[picked][0]
            st.session_state["drug_b"] = examples[picked][1]

    col1, col2 = st.columns(2)
    drug_a = col1.text_input("Drug A (name or DrugBank ID)", key="drug_a", placeholder="e.g. Warfarin or DB00682")
    drug_b = col2.text_input("Drug B (name or DrugBank ID)", key="drug_b", placeholder="e.g. Aspirin or DB00945")

    run = st.button("Predict interaction", type="primary", use_container_width=True)

    if run:
        predictor = load_predictor()
        with st.spinner("Running prediction..."):
            result = predictor.predict(drug_a, drug_b)

        if "error" in result:
            st.error(result["error"])
            return

        render_result(result)


if __name__ == "__main__":
    main()

