import os
from evaluate import load
from bert_score import score
from datasets import load_metric
import sacrebleu
import re
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate metrics for language models.")
parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the files.", default="./outputs/generations/")
parser.add_argument("--models", nargs='+', help="Model names to evaluate.")
parser.add_argument("--tests", nargs='+', default=['test_cl', 'test_other'], help="Tests to evaluate.")
parser.add_argument("--log_file_path", type=str, default='./outputs/logs/scores.log', help="Path to the log file.")
args = parser.parse_args()

# Initialize the METEOR and ROUGE metrics
meteor = load('meteor')
rouge = load('rouge')

# Use the command line arguments
base_dir = args.base_dir
model_sizes = args.model_sizes
tests = args.tests
log_file_path = args.log_file_path

# Create or clear the log file before writing the scores
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    log_file.write("METEOR, ROUGE, BLEU, and BERTScore Evaluation Results:\n")

def preprocess_text(text):
    # Remove specific tokens
    text = re.sub(r'\[INST\].*?\[/INST\]', '', text)
    text = text.replace("<pad>", "").replace("<s>", "").replace("</s>", "").replace("<unk>", "").strip()
    
    # Split text into sentences to remove any repeated ones
    sentences = text.split('.')
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence not in unique_sentences and sentence != "":
            unique_sentences.append(sentence)
    
    # Join the unique sentences back into a single string
    cleaned_text = '. '.join(unique_sentences).strip()
    if cleaned_text and not cleaned_text.endswith('.'):
        cleaned_text += '.'
    return cleaned_text

def evaluate_metrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        predictions = []
        references = []
        for line in file:
            if line.startswith('prediction:'):
                cleaned_prediction = preprocess_text(line.strip()[len('prediction:'):].strip())
                predictions.append(cleaned_prediction)  # Correct for METEOR, ROUGE, BERTScore
            elif line.startswith('reference:'):
                cleaned_reference = preprocess_text(line.strip()[len('reference:'):].strip())
                references.append(cleaned_reference)  # Correct for METEOR, ROUGE, BERTScore

    # Compute METEOR and ROUGE scores
    meteor_results = meteor.compute(predictions=predictions, references=references)
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # Compute BERTScore
    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    bert_score_result = F1.mean().item()  # Taking mean F1 score for simplicity

    # # Correct format for BLEU: references need to be a list of lists
    
    # bleu_predictions = [[pred] for pred in predictions]
    # bleu = load_metric('bleu')
    # bleu_results = bleu.compute(predictions=bleu_predictions, references=bleu_references)

    # Calculate BLEU score for all predictions
    bleu_references = [[ref] for ref in references]  # Adjust for BLEUs
    bleu_results = sacrebleu.corpus_bleu(predictions, bleu_references).score


    # Write the scores to the log file
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'Model: {file_path.replace(base_dir + "/", "")}\n')
        log_file.write(f'METEOR score: {meteor_results["meteor"]}\n')
        log_file.write(f'ROUGE scores: {rouge_results}\n')
        log_file.write(f'BLEU score: {bleu_results}\n')
        log_file.write(f'BERTScore: {bert_score_result}\n')
        log_file.write("\n")  # Add an empty line for better readability


# Iterate through the models, tests, and boolean conditions
for model_size in model_sizes:
    for test in tests:
        # generations of zero-shot models named as {model_name}-False
        for condition in ['False', 'True']:
            # Construct file path
            file_name = f'log-{test}-{model_size}-{condition}.txt'
            file_path = os.path.join(base_dir, file_name)
            
            try:
                # Check if the file exists
                if os.path.isfile(file_path):
                    evaluate_metrics(file_path)
                else:
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f'File does not exist: {file_path}\n\n')
            except Exception as e:
                print("skipping ", file_path)
                print(e)

print(f'Evaluation completed. Results are logged in {log_file_path}')
