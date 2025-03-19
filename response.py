import re
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_first_master_chef_response(conversation_text):

    # Split the conversation into turns
    turns = re.split(r"(Trainee Chef:|Master Chef:)", conversation_text)

    # Iterate through the turns to find the first Master Chef response
    speaker = None
    for i in range(1, len(turns)):  # Start from 1 to look at content after the speaker
        turn = turns[i].strip()
        if turns[i-1].strip() == "Master Chef:":
            # Found the first Master Chef response
            return turn

    return None  # If no Master Chef response is found

def calculate_bleu(reference, candidate):
    reference = [reference.split()]  # Must be list of lists
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

def calculate_bertscore(reference, candidate, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling

    ref_embedding = get_bert_embedding(reference)
    cand_embedding = get_bert_embedding(candidate)
    return cosine_similarity(ref_embedding, cand_embedding)[0][0]

# Specify the path to your text file
file_path = "paste.txt"  # Replace with the actual path to your text file

# Read the conversation text from the file
try:
    with open(file_path, "r", encoding="utf-8") as file:
        conversation_text = file.read()
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Define Your Reference Output
reference_output = """What a delicious-sounding dish! I'm excited to guide you through the preparation of Miso-Butter Roast Chicken With Acorn Squash Panzanella. Let's get started!

First, we need to pat the chicken dry with paper towels and season it all over with 2 tsp. kosher salt. You can use your hands or a spatula to gently massage the salt into the meat.

Next, tie the legs together with kitchen twine. This will help keep the chicken compact during roasting.

Finally, let the chicken sit at room temperature for about an hour. This step is important as it allows the skin to dry slightly and helps the seasonings penetrate deeper into the meat.

As you're doing this, I want to emphasize the importance of using a sharp knife when patting the chicken dry. If the blade is dull, it can tear the skin instead of simply removing excess moisture.

How's that going? Do you have any questions about seasoning the chicken or tying the legs together?"""  # Replace with your ground truth

# Extract the first Master Chef response
master_chef_output = extract_first_master_chef_response(conversation_text)

if master_chef_output:
    # Calculate metrics
    bleu_score = calculate_bleu(reference_output, master_chef_output)
    rouge_1, rouge_2, rouge_l = calculate_rouge(reference_output, master_chef_output)
    bertscore = calculate_bertscore(reference_output, master_chef_output)

    # Print the results
    print(f"Master Chef Output: {master_chef_output}")
    print(f"BLEU Score: {bleu_score}")
    print(f"ROUGE-1 F-score: {rouge_1}")
    print(f"ROUGE-2 F-score: {rouge_2}")
    print(f"ROUGE-L F-score: {rouge_l}")
    print(f"BERTScore: {bertscore}")
else:
    print("No Master Chef response found.")
