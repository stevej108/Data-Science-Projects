import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.workflow_labels import NODE_LABELS
import streamlit as st
from agents.run import run_workflow



st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide"
)


# ==========================================================
# Header
# ==========================================================

st.title("Enterprise Knowledge Assistant")

st.caption(
    "AI-powered document analysis and executive memo generation"
)


# ==========================================================
# Instructions
# ==========================================================

with st.expander("How to use this tool"):

    st.markdown(
        """
        ### Workflow

        This tool analyzes documents on your OneDrive and generates
        an executive memo containing a high level summary, and a summary of key findings from each relevant document.

        **Supported Documents**
        - PowerPoint (.pptx)
        - Excel (.xlsx)
        - Word (.docx)
        - PDF

        **Process**
        1. Adjust the OneDrive folder path to point to your OneDrive directory. (Change 'jsteve' to your username)
        2. Enter a business question or topic to search for.
        3. Adjust the maximum number of documents to summarize (default is 10).
        4. Set the output location for the generated memo (Change "Justin" to your name).
        5. Click "Generate Executive Memo" to start the analysis.

        **Note**
        - The tool will take approximately 10 minutes to complete the analysis, depending on the number of documents and their size.

        **Tips**
        - Use specific business questions
        - Include strategic objectives
        - Avoid overly broad searches
        """
    )


# ==========================================================
# Sidebar Configuration
# ==========================================================

with st.sidebar:

    st.header("Configuration")


    onedrive_root = st.text_input(
        "OneDrive Folder",
        value=r"C:\Users\jsteve\OneDrive - Burlington"
    )


    top_k = st.slider(
        "Maximum Documents",
        min_value=1,
        max_value=60,
        value=10
    )


    output_dir = st.text_input(
        "Memo Output Location",
        value=r"C:\Users\jsteve\OneDrive - Burlington\Documents\Document Search Memos\Justin"
    )


# ==========================================================
# Main Query
# ==========================================================

query = st.text_input(
    "Business Question",
    placeholder="Example: Summarize marketing transformation initiatives"
)


run_button = st.button(
    "Generate Executive Memo",
    type="primary"
)


# ==========================================================
# Workflow Execution
# ==========================================================

if run_button:

    if not query:

        st.warning(
            "Please enter a search topic."
        )

        st.stop()


    status = st.empty()

    progress_steps = [

        "Discovering Files",
        "Selecting Candidate Documents",
        "Parsing Documents",
        "Retrieving Relevant Content",
        "Ranking Sources",
        "Summarizing Documents",
        "Synthesizing Findings",
        "Creating Memo",

    ]


    try:

        status.info(
            "Starting workflow..."
        )


        progress = st.progress(0)

        status = st.empty()

        log_box = st.empty()

        logs = []

        steps = len(NODE_LABELS)

        completed = 0

        result = None

        for event in run_workflow(
            query=query,
            onedrive_root=onedrive_root,
            top_k=top_k,
            memo_output_path=output_dir,
        ):

            #
            # Progress update
            #
            if "__progress__" in event:

                icon, label = event["__progress__"]

                logs.append(f"{icon} {label}")

                progress.progress(
                    min(len(logs) / len(progress_steps), 1.0)
                )

                status.info(f"{icon} {label}")

                log_box.code("\n".join(logs))

                continue

            #
            # Final result
            #
            if "__final__" in event:

                result = event["__final__"]

                break


        status.success(
            "Workflow Complete- Memo Generated"
        )
        progress.progress(1.0)

        if result is None:
            st.error("Workflow did not return a final state.")
            st.stop()

        st.divider()


        # ==================================================
        # Executive Summary
        # ==================================================

        st.header(
            "Executive Summary"
        )

        st.write(
            result.get(
                "summary",
                "No summary generated."
            )
        )


        # ==================================================
        # Key Findings
        # ==================================================

        st.header(
            "Key Findings"
        )


        findings = result.get(
            "key_findings",
            []
        )


        for finding in findings:

            with st.container():

                st.subheader(
                    finding.get(
                        "title",
                        "Finding"
                    )
                )

                st.write(
                    finding.get(
                        "insight",
                        ""
                    )
                )


                if finding.get("business_impact"):

                    st.caption(
                        "Business Impact"
                    )

                    st.write(
                        finding["business_impact"]
                    )


        # ==================================================
        # Recommendations
        # ==================================================

        recommendations = result.get(
            "overall_recommendations",
            []
        )


        if recommendations:

            st.header(
                "Recommendations"
            )

            for rec in recommendations:

                st.write(
                    "•",
                    rec
                )


        # ==================================================
        # Memo
        # ==================================================

        st.header(
            "Generated Memo"
        )


        memo_path = result.get(
            "memo_path",
            ""
        )


        if memo_path:

            st.success(
                "Memo created successfully"
            )

            st.code(
                memo_path
            )


        # ==================================================
        # Debug / Logs
        # ==================================================

        with st.expander(
            "Workflow Log"
        ):

            for msg in result.get(
                "messages",
                []
            ):

                st.write(
                    "✓",
                    msg
                )


    except Exception as e:

        st.error(
            "Workflow failed"
        )

        st.exception(e)