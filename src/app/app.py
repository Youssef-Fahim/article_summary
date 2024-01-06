import os
import sys
import streamlit as st
# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..'))
from preprocessing.pdf_parsing import PDFParser
from model.transformer import Summarizer

def main():
    st.title("PDF Summarization App")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_parser = PDFParser()
        text = pdf_parser.parse_pdf_file(uploaded_file)
        summarizer = Summarizer()
        summary, summary_score = summarizer.summarize_text(text, "summary.txt")
        st.subheader("Summary")
        st.write(summary)

if __name__ == '__main__':
    main()