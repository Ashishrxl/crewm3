import os
import sys
import glob
import io
import re
import time
import logging
import queue
import threading
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos


# 1. Disable Telemetry before imports
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["POSTHOG_DISABLED"] = "true"

logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai.events").setLevel(logging.CRITICAL)

# --- Thread-Safe Queue Stream Writer ---
class QueueStream(io.StringIO):
    """
    Redirects stdout prints into a thread-safe Queue for real-time polling.
    """
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def write(self, s):
        if s:
            cleaned_str = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', s)
            if cleaned_str:
                self.log_queue.put(cleaned_str)
        return super().write(s)

    def flush(self):
        pass


# Configure Page
st.set_page_config(page_title="Parallel Deep Research Crew", layout="wide")

st.title("🔬 Parallel Deep Research Crew")
st.markdown("Automated research, fact-checking, and report creation powered by CrewAI & Gemini.")

gemini_key = st.secrets.get("GEMINI_API_KEY", "")
exa_key = st.secrets.get("EXA_API_KEY", "")

with st.sidebar:
    st.header("🔑 API Configuration")
    if not gemini_key:
        gemini_key = st.text_input("Enter Gemini API Key:", type="password")
    if not exa_key:
        exa_key = st.text_input("Enter EXA API Key:", type="password")

    st.info("API keys can also be saved in `.streamlit/secrets.toml` when deploying to Streamlit Cloud.")



def create_pdf(markdown_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)

    lines = markdown_text.split("\n")
    for line in lines:
        clean_line = line.strip().encode('latin-1', 'replace').decode('latin-1')

        # Heading 1
        if clean_line.startswith("# "):
            pdf.set_font("Helvetica", style="B", size=18)
            pdf.cell(
                0, 
                10, 
                text=clean_line.replace("# ", "").strip(), 
                new_x=XPos.LMARGIN, 
                new_y=YPos.NEXT
            )
            pdf.ln(2)
        # Heading 2
        elif clean_line.startswith("## "):
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(
                0, 
                8, 
                text=clean_line.replace("## ", "").strip(), 
                new_x=XPos.LMARGIN, 
                new_y=YPos.NEXT
            )
            pdf.ln(2)
        # Heading 3
        elif clean_line.startswith("### "):
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(
                0, 
                6, 
                text=clean_line.replace("### ", "").strip(), 
                new_x=XPos.LMARGIN, 
                new_y=YPos.NEXT
            )
            pdf.ln(1)
        # Body text / List items
        else:
            text_line = clean_line.replace("**", "").replace("__", "")
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, text=text_line)
            pdf.ln(1)

    return bytes(pdf.output())


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
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["EXA_API_KEY"] = exa_key

        from crew import ParallelDeepResearchCrew

        status_box = st.status("🚀 Running Deep Research Crew...", expanded=True)
        log_container = status_box.empty()

        # Thread-safe queue to pass log strings from background thread to Streamlit
        log_queue = queue.Queue()
        accumulated_logs = []

        # Container for execution results across threads
        execution_result = {"result": None, "error": None}

        def run_crew_in_thread():
            """Worker thread function to execute CrewAI without blocking UI."""
            sys_stdout_orig = sys.stdout
            sys.stdout = QueueStream(log_queue)
            try:
                crew_obj = ParallelDeepResearchCrew().crew()
                execution_result["result"] = crew_obj.kickoff(inputs={"user_query": user_query})
            except Exception as thread_err:
                execution_result["error"] = thread_err
            finally:
                sys.stdout = sys_stdout_orig

        # Start background thread
        crew_thread = threading.Thread(target=run_crew_in_thread)
        crew_thread.start()

        # --- Real-Time Polling Loop on Main Thread ---
        while crew_thread.is_alive() or not log_queue.empty():
            updated = False
            while not log_queue.empty():
                log_chunk = log_queue.get()
                accumulated_logs.append(log_chunk)
                updated = True

            # Update Streamlit code block in real time
            if updated and accumulated_logs:
                full_log_text = "".join(accumulated_logs)
                # Keep last 2500 chars visible during live execution to prevent lag
                visible_log = full_log_text[-2500:] if len(full_log_text) > 2500 else full_log_text
                log_container.code(visible_log, language="bash")

            time.sleep(0.1)

        crew_thread.join()

        # Handle Execution Results
        if execution_result["error"]:
            status_box.update(label="❌ Execution Failed!", state="error", expanded=True)
            st.error(f"An error occurred during execution: {str(execution_result['error'])}")
            if accumulated_logs:
                st.subheader("📋 Error Logs")
                st.code("".join(accumulated_logs), language="bash")
        else:
            status_box.update(label="✅ Research Execution Complete!", state="complete", expanded=False)
            st.success("Research completed successfully!")

            result = execution_result["result"]
            report_content = ""
            if os.path.exists("final_report.md"):
                with open("final_report.md", "r", encoding="utf-8") as f:
                    report_content = f.read()
            else:
                report_content = str(result)

            st.subheader("📄 Final Research Report")
            st.markdown(report_content)

            # Full Log View
            with st.expander("📋 View Full Real-Time Console Logs", expanded=False):
                st.code("".join(accumulated_logs), language="bash")

            # Downloads
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

            # Visualizations
            plot_files = glob.glob("plots/*.png")
            if plot_files:
                st.markdown("---")
                st.subheader("📊 Generated Visualizations")
                cols = st.columns(min(len(plot_files), 2))
                for idx, plot_path in enumerate(plot_files):
                    col = cols[idx % 2]
                    col.image(plot_path, use_container_width=True)
