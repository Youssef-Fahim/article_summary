import os
import sys
import re
import PyPDF2
#import pytesseract
#from PIL import Image

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'utils'))

from logger import get_logger

APP_NAME = 'pdf_parsing'
LOGGER = get_logger(APP_NAME)

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + '/data'
PDF_DIR = DATA_DIR + '/pdf_files'
TEXT_DIR = DATA_DIR + '/text_files'
OCR_DIR = DATA_DIR + '/ocr_files'

#print(sys.path)
print(os.listdir(PDF_DIR))

class PDFParser:
    def __init__(self, pdf_dir=PDF_DIR, text_dir=TEXT_DIR, logger=LOGGER):
        self.pdf_dir = pdf_dir
        self.text_dir = text_dir
        self.logger = logger

    def clean_text_directory(self):
        # Number of files in text directory
        num_files = len(os.listdir(self.text_dir))
        self.logger.info(f'Number of files in text directory: {num_files}')
        for file_name in os.listdir(self.text_dir):
            if file_name.endswith('.txt'):
                os.remove(os.path.join(self.text_dir, file_name))
                self.logger.info(f'Removed file: {file_name}')

    def parse_pdf_file(self, pdf_file):
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        # Get number of pages
        num_pages = len(pdf_reader.pages)
        self.logger.info(f'Number of pages in {pdf_file}: {num_pages}')
        # Initialize page text
        page_text = ''
        # Iterate through all pages
        for page_num in range(num_pages):
            # Get page object
            page = pdf_reader.pages[page_num]
            # Extract text from page
            page_text += page.extract_text()

        return page_text
       

    def save_text_to_directory(self, text, text_file_name):
        # Create text file path
        text_file_path = os.path.join(self.text_dir, text_file_name)
        # Create text file
        with open(text_file_path, 'w') as text_file:
            # Write text to text file
            text_file.write(text)
            self.logger.info(f'Wrote text to file: {text_file_name}')

    def remove_irrelevant_infos(self, text):
        # Remove web_page address
        # URLs don't contribute to the semantic meaning of the text
        text = re.sub(r'www\.\S+\.\w{2,3}', '', text)
        # Remove email address
        text = re.sub(r'\S+@\S+\.\w{2,3}', '', text)
        # Remove phone number
        text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)
        return text
    
    def remove_punctation_errors(self, text):
        # Remove white spaces before punctation
        # e.g. 'Hello , my name is John' -> 'Hello, my name is John'
        text = re.sub(r'\s([,.!?])', r'\1', text)
        # Remove white spaces before and after apostrophe
        # e.g. 'I ' m' -> 'I'm'
        text = re.sub(r'\s\'\s', '\'', text)
        return text

    def text_preprocessing(self, text):
        # Remove irrelevant infos
        text = self.remove_irrelevant_infos(text)
        # Remove punctation errors
        text = self.remove_punctation_errors(text)
        return text
        

    def parse_pdfs_from_local_folder(self):
        # Clean text directory
        self.clean_text_directory()
        # Iterate through all PDF files
        for file_name in os.listdir(self.pdf_dir):
            if file_name.endswith('.pdf'):
                self.logger.info(f'Parsing PDF file: {file_name}')
                # Open PDF file
                with open(os.path.join(self.pdf_dir, file_name), 'rb') as pdf_file:
                    # Create text file name
                    text_file_name = file_name.replace('.pdf', '.txt')
                    # Parse PDF file
                    text = self.parse_pdf_file(pdf_file)
                    # Preprocess text
                    text = self.text_preprocessing(text)
                    # Save text to directory
                    self.save_text_to_directory(text, text_file_name)
                    

def main():
    pdf_parser = PDFParser()
    pdf_parser.parse_pdfs_from_local_folder()

if __name__ == '__main__':
    main()
