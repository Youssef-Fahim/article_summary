# Path: src/model/transformer.py
import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'utils'))
from logger import get_logger
from config import read_params

# Read parameters
CONFIG_DIR = os.path.join(SCRIPT_DIR, '..', '..' ,'config')
params = read_params(CONFIG_DIR)
# Set parameters
MODEL_PATH = params['MODEL']['model_path']
MODEL_INPUT_MAX_LENGTH = params['MODEL']['model_input_max_length']
GAMMA = params['MODEL']['gamma']
# Set logger
APP_NAME = 'Transformer_Summarizer'
LOGGER = get_logger(APP_NAME)
# Set directories
TEXT_DIR = 'data/text_files'
SUMMARY_DIR = 'data/summary_files'


class Summarizer:
    def __init__(self, 
                model_path=MODEL_PATH, 
                text_dir=TEXT_DIR, 
                batch_length=MODEL_INPUT_MAX_LENGTH, 
                max_length_summary=int(MODEL_INPUT_MAX_LENGTH * GAMMA),
                summary_dir=SUMMARY_DIR,
                logger=LOGGER):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_dir = text_dir
        self.batch_length = batch_length
        self.max_length_summary = max_length_summary
        self.summary_dir = summary_dir
        self.logger = logger

    def compute_summary_and_score(self, input_ids, attention_mask):
        # Add batch dimension
        # The model expects input tensors to be of shape [batch_size, sequence_length]
        # Therefore, we need to add a batch dimension to our input tensors
        input_ids = input_ids.unsqueeze(0) # shape [1, sequence_length]
        attention_mask = attention_mask.unsqueeze(0) # shape [1, sequence_length]
        # Generate summary
        summary_ids = self.model.generate(input_ids, attention_mask=attention_mask, 
                                          max_length=self.max_length_summary,
                                          num_beams=4, early_stopping=True, output_scores=True,
                                          return_dict_in_generate=True)
        # Decode summary
        summary = self.tokenizer.batch_decode(summary_ids[0], skip_special_tokens=True,
                                             remove_invalid_values=True)
        # log summary
        self.logger.info(f'Summary: {summary[0]}')
        # Get summary score
        summary_score = summary_ids.sequences_scores.numpy()[0]
        # log summary score
        self.logger.info(f'Summary score: {summary_score}')
        # Return summary and summary score
        return summary, summary_score

    def compute_batch_summary_and_score(self, input_text, batch_length):
        # Encode text
        encoded_input = self.tokenizer(input_text, padding='max_length', return_tensors='pt', 
                                       max_length=self.batch_length, truncation=True,
                                       return_overflowing_tokens=True)
        # encoded_input = self.tokenizer(input_text, return_tensors='pt', 
        #                                max_length=self.batch_length, truncation=True)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        summaries = []
        summary_scores = []
        # Number of batches
        num_batches = len(input_ids)
        self.logger.info(f'Number of batches: {num_batches}')
        # Iterate through batches
        for inp_ids, att_mask in zip(input_ids, attention_mask):
            self.logger.info(f'Batch length: {len(inp_ids)}')
            # Compute summary and summary score
            summary, summary_score = self.compute_summary_and_score(inp_ids, att_mask)
            # Append summary and summary score
            summaries.append(summary[0])
            summary_scores.append(summary_score)
        # Return summary and summary score
        return summaries, summary_scores
    
    def concatenate_summaries(self, list_of_summaries):
        # Concatenate summaries
        concatenated_summary = '\n'.join(list_of_summaries)
        # Return concatenated summary
        return concatenated_summary
    
    def compute_mean_summary_score(self, list_of_summary_scores):
        # Compute mean summary score
        mean_summary_score = sum(list_of_summary_scores) / len(list_of_summary_scores)
        # Return mean summary score
        return mean_summary_score
    
    def save_summary_to_directory(self, summary, file_name):
        # Create summary file path
        summary_file_path = os.path.join(self.summary_dir, file_name)
        # Create summary file
        with open(summary_file_path, 'w') as summary_file:
            # Write summary to summary file
            summary_file.write(summary)

    def summarize_text(self, text, text_file_name):
        # Compute summary and summary score
        summaries, summary_scores = self.compute_batch_summary_and_score(text, self.batch_length)
        # Print summary
        concatenated_summary = self.concatenate_summaries(summaries)
        # Print summary score
        mean_summary_score = self.compute_mean_summary_score(summary_scores)

        return concatenated_summary, mean_summary_score

    
    def summarize_from_local_folder(self):
        # Iterate through all text files
        for text_file_name in os.listdir(self.text_dir):
            # Log text file name
            self.logger.info(f'Text file name: {text_file_name}')
            if text_file_name.endswith('.txt'):
                # Create text file path
                text_file_path = os.path.join(self.text_dir, text_file_name)
                # Open text file
                with open(text_file_path, 'r') as text_file:
                    # Read text file
                    text = text_file.read()
                    # Summarize text
                    concatenated_summary, mean_summary_score = \
                        self.summarize_text(text, text_file_name)
                    print(concatenated_summary)
                    print(mean_summary_score)
                    # Save summary to directory
                    summary_text_file_name = text_file_name.replace('.txt', '_summary.txt')
                    self.save_summary_to_directory(concatenated_summary, summary_text_file_name)

def main():
    summarizer = Summarizer()
    summarizer.summarize_from_local_folder()

if __name__ == '__main__':
    main()