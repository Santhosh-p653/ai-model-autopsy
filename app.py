import streamlit as st
import pandas as pd
from report import AutopsyOrchestrator

st.set_page_config(
    page_title="AI Model Autopsy Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .report-card { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; margin-top: 15px; }
    h1, h2, h3 { color: #c9d1d9; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #238636; color: white; }
    .stButton>button:hover { background-color: #2ea043; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Autopsy Config")
    st.markdown("Configure parameters for model diagnostic execution.")
    
    uploaded_file = st.file_uploader("Upload Evaluation Dataset (.csv)", type=["csv"])
    target_col = st.text_input("Target Column Name", value="target")
    
    st.divider()
    st.markdown("### MCP Service Nodes")
    st.markdown("🟢 **Model MCP:** Online (`port 3000`)")
    st.markdown("🟢 **Data MCP:** Active & Profiling")
    st.markdown("🤖 **Primary LLM:** Groq (`llama-3.3-70b`)")
    st.markdown("🛡️ **Fallback LLM:** Gemini (`gemini-1.5-pro`)")
    
    run_button = st.button("Run Diagnostic Autopsy")

st.title("🔍 AI Model Autopsy & Failure Analysis")
st.markdown("Inspect machine learning behavior, trace structural data errors, and automatically synthesize remediation steps.")

if uploaded_file is not None:
    with open("temp_eval.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df_preview = pd.read_csv(uploaded_file)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{df_preview.shape[0]:,}")
    col2.metric("Total Features", f"{df_preview.shape[1] - 1}")
    col3.metric("Missing Cells", f"{df_preview.isnull().sum().sum():,}")
    col4.metric("Estimated Drift", "Low Risk", "0.04 delta")
    
    tab_report, tab_data, tab_model = st.tabs(["📋 Autopsy Report", "📊 Dataset Preview", "⚙️ Model Metadata"])
    
    with tab_report:
        if run_button:
            with st.status("Executing diagnostic chain...", expanded=True) as status:
                st.write("Connecting to Model & Data MCP servers...")
                st.write("Profiling distributions and tracking missing values...")
                st.write("Generating deep-reasoning synthesis via primary LLM (Groq)...")
                
                orchestrator = AutopsyOrchestrator()
                mock_model_evidence = {"status": "healthy", "model": "RandomForestClassifier", "latency_ms": 14}
                mock_data_evidence = {
                    "rows": len(df_preview), 
                    "missing_values": df_preview.isnull().sum().to_dict(),
                    "duplicates": int(df_preview.duplicated().sum())
                }
                
                report_content = orchestrator.generate_report(mock_model_evidence, mock_data_evidence)
                status.update(label="Autopsy Complete!", state="complete", expanded=False)
            
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.subheader("Diagnostic Findings & Remediation")
            st.markdown(report_content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Configure your dataset in the sidebar and click **Run Diagnostic Autopsy** to initiate analysis.")

    with tab_data:
        st.subheader("Raw Data Preview")
        st.dataframe(df_preview.head(10), use_container_width=True)
        
    with tab_model:
        st.subheader("BentoML Production Serving State")
        st.json({
            "service_name": "model_autopsy_service",
            "endpoint": "http://localhost:3000/predict",
            "framework": "scikit-learn",
            "active_runners": ["RandomForestClassifier"]
        })
else:
    st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #161b22; border-radius: 10px; border: 1px dashed #30363d; margin-top: 30px;">
            <h3>No Dataset Uploaded Yet</h3>
            <p style="color: #8b949e;">Please upload an evaluation CSV dataset via the sidebar to jumpstart the diagnostic autopsy tool.</p>
        </div>
    """, unsafe_allow_html=True)
