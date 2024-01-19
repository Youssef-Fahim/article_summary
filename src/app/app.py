import os
import sys
import base64
import streamlit as st
from streamlit_javascript import st_javascript

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..'))
from preprocessing.pdf_parsing import PDFParser
from model.transformer import Summarizer

def display_pdf(uploaded_file, width):
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()

    # Convert bytes data to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # Embed base64 encoded PDF file into HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width={str(width)} height={str(width*4/3)} type="application/pdf"></iframe>'

    # Display PDF
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide")
    st.title("PDF Summarization App")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a PDF file", 
                                         type="pdf",
                                         help="Only PDF files are allowed")

    if uploaded_file is not None:
        pdf_parser = PDFParser()
        text = pdf_parser.parse_pdf_file(uploaded_file)
        summarizer = Summarizer()
        summary, summary_score = summarizer.summarize_text(text, "summary.txt")
        
        # Create 2 columns
        col1, col2 = st.columns(spec=[2, 1], gap="small")

        # Use the first column to display the PDF file
        with col1:
            ui_width = st_javascript("window.innerWidth")
            st.header("PDF File")
            display_pdf(uploaded_file, ui_width - 10)

        # Use the second column to display the summary
        with col2:
            st.header("Summary")
            st.write(summary)

if __name__ == '__main__':
    main()