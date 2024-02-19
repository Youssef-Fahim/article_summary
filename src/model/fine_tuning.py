""" 
Description: This script is used to fine-tune the model on the scientific papers dataset.
"""

# Import libraries
import os
import sys
import torch
from transformers import (Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments, 
                          AutoTokenizer, 
                          BartTokenizer,
                          AutoModelForSeq2SeqLM,
                          BartForConditionalGeneration
                          )
from datasets import load_dataset, load_metric

# Custom library imports
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from utils.logger import get_logger
from utils.config import read_params

# Set logger
APP_NAME = 'model_fine_tuning'
LOGGER = get_logger(APP_NAME)

# In case operator not available for MPS, use CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set device
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
LOGGER.info(f'Device: {DEVICE}')

# Read parameters
CONFIG_DIR = os.path.join(SCRIPT_DIR, '..', '..' ,'config')
params = read_params(CONFIG_DIR)

# Set parameters
MODEL_PATH = params['MODEL']['model_path']
MAX_INPUT_LENGTH = params['MODEL']['model_input_max_length']
MAX_OUTPUT_LENGTH = params['MODEL']['model_output_max_length']
batch_size = 2


class ModelFineTuning():

    def __init__(self, 
                 model_path=MODEL_PATH, 
                 max_input_length=MAX_INPUT_LENGTH, 
                 max_output_length=MAX_OUTPUT_LENGTH,
                 batch_size=batch_size):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def compute_train_val_test_data(self):
        # Load the dataset
        dataset = load_dataset('scientific_papers', 'arxiv')

        # Get the training, validation and test datasets
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['validation']
        self.test_dataset = dataset['test']

        LOGGER.info("Train/Val/Test Datasets loaded successfully.")

    def process_data_to_model_inputs(self, batch):    
        # tokenize the inputs and labels
        inputs = self.tokenizer(batch["article"], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=MAX_INPUT_LENGTH)
        
        outputs = self.tokenizer(batch["abstract"], 
                            padding="max_length", 
                            truncation=True, 
                            max_length=MAX_OUTPUT_LENGTH)
        
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids

        # ignore the PAD token
        batch["labels"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in
                        batch["labels"]]
        
        return batch

    def encode_and_format_data(self, dataset):
        dataset = dataset.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["article", "abstract", "section_names"],
        )

        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    def compute_metrics(self, pred):
        rouge = load_metric("rouge")
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids==-100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"] 
        )["rouge2"].mid 

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4)
        }
    
    def set_model_hyperparameters(self):
        # set hyperparameters
        self.model.config.num_beams = 2
        self.model.config.max_length = 128
        self.model.config.min_length = 80
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3

    def save_model(self, hub=False):
        # Evaluate the model
        metrics = self.trainer.evaluate()
        # Access eval_rouge2_fmeasure and epoch
        eval_rouge2_fmeasure = metrics['eval_rouge2_fmeasure']
        epoch = metrics["epoch"]
        # Save the model to Hugging Face Hub
        save_model_path = f"bart_fine_tuned_{epoch}_epochs_rouge2_{eval_rouge2_fmeasure:.3f}"
        if not hub:
            # Save the model locally
            self.trainer.save_model(save_model_path)
            LOGGER.info(f"Model saved locally to: {save_model_path}")
        else:
            # Save the model to Hugging Face Hub
            self.trainer.push_to_hub(save_model_path)
            LOGGER.info(f"Model saved to Hugging Face Hub: {save_model_path}")


    def fine_tune(self):

        self.compute_train_val_test_data()

        self.tokenizer = BartTokenizer.from_pretrained(self.model_path)

        train_sample = 150 #250
        val_sample = 5 #25
        self.train_dataset = self.train_dataset.select(range(train_sample))
        self.val_dataset = self.val_dataset.select(range(val_sample))

        self.train_dataset = self.encode_and_format_data(self.train_dataset)
        self.val_dataset = self.encode_and_format_data(self.val_dataset)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, 
                                                           gradient_checkpointing=True, 
                                                           use_cache=False)

        # set hyperparameters
        self.set_model_hyperparameters()

        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            #fp16=True,
            use_mps_device=True,
            output_dir="./",
            logging_steps=5,
            eval_steps=10,
            save_steps=10,
            save_total_limit=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
        )

        self.trainer = Seq2SeqTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
            )

        self.trainer.train()

        self.save_model(hub=True)

if __name__ == "__main__":
    model_finetuning = ModelFineTuning()
    model_finetuning.fine_tune()
    LOGGER.info("Model fine-tuning completed successfully.")