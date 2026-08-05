import os
import streamlit as st
import glob

# Configure Page
st.set_page_config(page_title="Parallel Deep Research Crew", layout="wide")

st.title("🔬 Parallel Deep Research Crew")
st.markdown("Automated research, fact-checking, and report creation powered by CrewAI & Gemini.")

# Load Keys from Streamlit Secrets or Sidebar Input
gemini_key = st.secrets.get("GEMINI_API_KEY", "")
exa_key = st.secrets.get("EXA_API_KEY", "")

with st.sidebar:
    st.header("API Configuration")
    if not gemini_key:
        gemini_key = st.text_input("Enter Gemini API Key:", type="password")
    if not exa_key:
        exa_key = st.text_input("Enter EXA API Key:", type="password")
    
    st.info("API keys can also be saved in `.streamlit/secrets.toml` when deploying to Streamlit Cloud.")

# Input Query
user_query = st.text_area("Enter your research topic/query:", height=100, placeholder="e.g., Trends in Renewable Energy Adoption 2024-2026")

if st.button("Start Deep Research", type="primary"):
    if not gemini_key or not exa_key:
        st.error("Please provide both GEMINI_API_KEY and EXA_API_KEY.")
    elif not user_query.strip():
        st.warning("Please enter a research topic.")
    else:
        # Set Environment Variables for CrewAI & EXA
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["EXA_API_KEY"] = exa_key
        
        # Import Crew dynamically after env vars are populated
        from crew import ParallelDeepResearchCrew

        with st.spinner("Researching, validating data, and writing final report..."):
            try:
                # Run the Crew process
                crew_obj = ParallelDeepResearchCrew().crew()
                result = crew_obj.kickoff(inputs={"user_query": user_query})
                
                st.success("Research completed successfully!")
                
                # Render Final Report
                st.subheader("📄 Final Research Report")
                if os.path.exists("final_report.md"):
                    with open("final_report.md", "r", encoding="utf-8") as f:
                        report_content = f.read()
                    st.markdown(report_content)
                else:
                    st.markdown(str(result))

                # Render Generated Plots if created by ChartGeneratorTool
                plot_files = glob.glob("plots/*.png")
                if plot_files:
                    st.subheader("📊 Generated Visualizations")
                    cols = st.columns(min(len(plot_files), 2))
                    for idx, plot_path in enumerate(plot_files):
                        col = cols[idx % 2]
                        col.image(plot_path, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during execution: {str(e)}")
