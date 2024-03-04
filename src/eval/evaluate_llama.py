import argparse
from datasets import load_metric, load_from_disk
import sacrebleu
from transformers import pipeline
from bert_score import score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

def main(task, base_model_name, local_model_to_evaluate_path, hf_model_to_evaluate, test_dataset_folder, log_file):
    
    base_model = f"NousResearch/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model)  # Load base model
    if local_model_to_evaluate_path:
        model_state_dict = torch.load(os.path.join(local_model_to_evaluate_path, "model.pkl"), map_location='cpu')
        model.load_state_dict(model_state_dict)
    elif hf_model_to_evaluate:
        model = AutoModelForCausalLM.from_pretrained(hf_model_to_evaluate)

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset_path = f"{test_dataset_folder}_{task}"
    test_dataset = load_from_disk(test_dataset_path)

    # Prepare the pipeline
    pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Initialize metrics
    bleu_metric = load_metric("sacrebleu")
    bert_scores = []
    bleu_scores= []

    # Function for generating predictions with truncation
    def generate_prediction(input_ids, attention_mask):
        # Truncate input_ids and attention_mask to model's max length
        input_ids = input_ids[:512]
        attention_mask = attention_mask[:512]

        # Decode input_ids to text
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Generate outputs using text input
        outputs = pipe(input_text, max_new_tokens=512)
        return outputs[0]['generated_text'] if outputs else None

    predictions = []
    references  = []

    # Evaluate the dataset
    for data in test_dataset:
        try:
            # Generate prediction
            prediction = generate_prediction(data['input_ids'], data['attention_mask'])

            # Decode the reference
            reference = tokenizer.decode(data['labels'])
            reference = re.sub(r'\[INST\].*?\[/INST\]', '', reference)
            print("reference:", reference)
            log_file.write(f"reference: {reference}\n")
            print("prediction:", prediction)
            log_file.write(f"prediction: {prediction}\n")

        # Append the prediction and reference to their respective lists
            prediction = prediction.replace("<pad>", "")
            prediction = prediction.replace("<unk>", "")
            reference = reference.replace("<pad>", "")
            reference = reference.replace("<unk>", "")
            reference = reference.replace("</s>", "")

            predictions.append(prediction)
            references.append([reference])  # Note that references need to be a list of lists

            # Compute BERTScore
            P, R, F1 = score([prediction], [reference], lang="en")
            bert_scores.append(F1.mean().item())

        except Exception as e:
            print(f"Error processing data: {e}")

    # Make sure predictions are generated
    if not predictions:
        print("No predictions were generated.")
        log_file.write("No predictions were generated.")
    else:
        # Calculate BLEU score for all predictions
        bleu_score = sacrebleu.corpus_bleu(predictions, references).score

        # Calculate average BERTScore
        average_bert_score = sum(bert_scores) / len(bert_scores) if bert_scores else 0

        print(f"BLEU score: {bleu_score}")
        log_file.write(f"BLEU score: {bleu_score}\n")
        print(f"Average BERTScore: {average_bert_score}")
        log_file.write(f"Average BERTScore: {average_bert_score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=str, help='Task type: test_cl or test_other')
    parser.add_argument('--base_model_name', type=str, help='Model name: Huggingface id of Llama-2 family')
    parser.add_argument('--local_model_to_evaluate_path', type=str, help='Local path of trained model with intermediate task pre-training',default=None)
    parser.add_argument('--hf_model_to_evaluate', type=str, help='Huggingface id of trained model with intermediate task pre-training', default=None)
    parser.add_argument('--test_dataset_folder', type=str, help='Training dataset path under data', default='./data/llama_compliant_hf_')
    parser.add_argument('--log_dir', type=str, help='Directory to log results', default="./outputs/generations/")
    args = parser.parse_args()

    evaluate_model_str = None
    if not args.local_model_to_evaluate_path:
        evaluate_model_str = args.local_model_to_evaluate_path
    elif not args.hf_model_to_evaluate:
        evaluate_model_str = args.hf_model_to_evaluate

    # Open log file
    with open(f"{args.log_dir}-{args.task}-{args.model}-{evaluate_model_str}.txt", "a") as log_file:
        # Log the arguments
        print("Arguments:", args)
        log_file.write(f"Arguments: {args}\n")

        # Check if the task argument starts with 'test'
        if args.task.startswith('test'):
            main(args.task, args.base_model_name, 
                 args.local_model_to_evaluate_path, 
                 args.hf_model_to_evaluate, 
                 args.test_dataset_folder, 
                 log_file)
        else:
            error_message = "Task must start with 'test'"
            print(error_message)
            log_file.write(error_message + "\n")
