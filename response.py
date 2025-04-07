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
file_path = "cleaned_conversation_test.txt"  # Replace with the actual path to your text file

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
reference_output = """Meanwhile, halve squash and scoop out seeds. Run a vegetable peeler along ridges of squash halves to remove skin. Cut each half into ½""-thick wedges; arrange on a rimmed baking sheet.
Combine sage, rosemary, and 6 Tbsp. melted butter in a large bowl; pour half of mixture over squash on baking sheet. Sprinkle squash with allspice, red pepper flakes, and ½ tsp. salt and season with black pepper; toss to coat.
Add bread, apples, oil, and ¼ tsp. salt to remaining herb butter in bowl; season with black pepper and toss to combine. Set aside.
Place onion and vinegar in a small bowl; season with salt and toss to coat. Let sit, tossing occasionally, until ready to serve.
Place a rack in middle and lower third of oven; preheat to 425°F. Mix miso and 3 Tbsp. room-temperature butter in a small bowl until smooth. Pat chicken dry with paper towels, then rub or brush all over with miso butter. Place chicken in a large cast-iron skillet and roast on middle rack until an instant-read thermometer inserted into the thickest part of breast registers 155°F, 50–60 minutes. (Temperature will climb to 165°F while chicken rests.) Let chicken rest in skillet at least 5 minutes, then transfer to a plate; reserve skillet.
Meanwhile, roast squash on lower rack until mostly tender, about 25 minutes. Remove from oven and scatter reserved bread mixture over, spreading into as even a layer as you can manage. Return to oven and roast until bread is golden brown and crisp and apples are tender, about 15 minutes. Remove from oven, drain pickled onions, and toss to combine. Transfer to a serving dish.
Using your fingers, mash flour and butter in a small bowl to combine.
Set reserved skillet with chicken drippings over medium heat. You should have about ¼ cup, but a little over or under is all good. (If you have significantly more, drain off and set excess aside.) Add wine and cook, stirring often and scraping up any browned bits with a wooden spoon, until bits are loosened and wine is reduced by about half (you should be able to smell the wine), about 2 minutes. Add butter mixture; cook, stirring often, until a smooth paste forms, about 2 minutes. Add broth and any reserved drippings and cook, stirring constantly, until combined and thickened, 6–8 minutes. Remove from heat and stir in miso. Taste and season with salt and black pepper.
Serve chicken with gravy and squash panzanella alongside."""  # Replace with your ground truth

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
