import os
import sys
import glob
import io
import re
import logging
import streamlit as st
from fpdf import FPDF

# 1. Disable CrewAI Telemetry & Tracing
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["POSTHOG_DISABLED"] = "true"

# 2. Suppress noisy telemetry and event loggers
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai.events").setLevel(logging.CRITICAL)

# Configure Page
st.set_page_config(page_title="Parallel Deep Research Crew", layout="wide")

st.title("🔬 Parallel Deep Research Crew")
st.markdown("Automated research, fact-checking, and report creation powered by CrewAI & Gemini.")

# Load Keys from Streamlit Secrets or Sidebar Input
gemini_key = st.secrets.get("GEMINI_API_KEY", "")
exa_key = st.secrets.get("EXA_API_KEY", "")

with st.sidebar:
    st.header("🔑 API Configuration")
    if not gemini_key:
        gemini_key = st.text_input("Enter Gemini API Key:", type="password")
    if not exa_key:
        exa_key = st.text_input("Enter EXA API Key:", type="password")

    st.info("API keys can also be saved in `.streamlit/secrets.toml` when deploying to Streamlit Cloud.")


class StreamToStreamlit(io.StringIO):
    """
    Custom stream capture class that strips ANSI color codes 
    and updates a Streamlit code container live.
    """
    def __init__(self, st_container):
        super().__init__()
        self.st_container = st_container
        self.buffer = []

    def write(self, s):
        # Strip ANSI control sequences (colors, formatting)
        cleaned_str = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', s)
        self.buffer.append(cleaned_str)
        # Update Streamlit code widget with full accumulated log buffer
        self.st_container.code("".join(self.buffer), language="bash")

    def flush(self):
        pass


def create_pdf(markdown_text: str) -> bytes:
    """
    Converts plain markdown text into a clean PDF document byte string using fpdf2.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)

    # Process text line by line to handle headers and body text cleanly
    lines = markdown_text.split("\n")
    for line in lines:
        clean_line = line.strip()

        # Sanitize text for standard Latin-1 PDF encoding
        clean_line = clean_line.encode('latin-1', 'replace').decode('latin-1')

        # Heading 1
        if clean_line.startswith("# "):
            pdf.set_font("Helvetica", style="B", size=18)
            pdf.cell(0, 10, txt=clean_line.replace("# ", "").strip(), ln=True)
            pdf.ln(2)
        # Heading 2
        elif clean_line.startswith("## "):
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(0, 8, txt=clean_line.replace("## ", "").strip(), ln=True)
            pdf.ln(2)
        # Heading 3
        elif clean_line.startswith("### "):
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(0, 6, txt=clean_line.replace("### ", "").strip(), ln=True)
            pdf.ln(1)
        # Body text / List items
        else:
            # Strip simple bold markdown stars for basic rendering
            text_line = clean_line.replace("**", "").replace("__", "")
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, txt=text_line)
            pdf.ln(1)

    return bytes(pdf.output())


# User Query Input
user_query = st.text_area(
    "Enter your research topic/query:", 
    height=100, 
    placeholder="e.g., Impact of Quantum Computing on Financial Cryptography in 2026"
)

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

        # Streamlit status container to show live execution steps
        status_box = st.status("🚀 Running Deep Research Crew...", expanded=True)
        log_container = status_box.empty()

        # Redirect standard output (sys.stdout) to custom Streamlit stream
        sys_stdout_orig = sys.stdout
        sys.stdout = StreamToStreamlit(log_container)

        try:
            # Execute Crew process
            crew_obj = ParallelDeepResearchCrew().crew()
            result = crew_obj.kickoff(inputs={"user_query": user_query})

            status_box.update(label="✅ Research Execution Complete!", state="complete", expanded=False)
            st.success("Research completed successfully!")

            # Retrieve report content
            report_content = ""
            if os.path.exists("final_report.md"):
                with open("final_report.md", "r", encoding="utf-8") as f:
                    report_content = f.read()
            else:
                report_content = str(result)

            # Render Final Report
            st.subheader("📄 Final Research Report")
            st.markdown(report_content)

            # --- Download Buttons Section ---
            st.markdown("---")
            st.subheader("📥 Download Report")
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📄 Download Markdown (.md)",
                    data=report_content,
                    file_name="final_research_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with col2:
                try:
                    pdf_data = create_pdf(report_content)
                    st.download_button(
                        label="📕 Download PDF (.pdf)",
                        data=pdf_data,
                        file_name="final_research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.warning(f"Unable to generate PDF: {str(pdf_err)}")

            # Render Generated Plots if created by ChartGeneratorTool
            plot_files = glob.glob("plots/*.png")
            if plot_files:
                st.markdown("---")
                st.subheader("📊 Generated Visualizations")
                cols = st.columns(min(len(plot_files), 2))
                for idx, plot_path in enumerate(plot_files):
                    col = cols[idx % 2]
                    col.image(plot_path, use_container_width=True)

        except Exception as e:
            status_box.update(label="❌ Execution Failed!", state="error", expanded=True)
            st.error(f"An error occurred during execution: {str(e)}")

        finally:
            # Always restore standard stdout after execution completes or fails
            sys.stdout = sys_stdout_orig
