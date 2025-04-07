import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np

def read_conversation(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def separate_responses(conversation):
    chef_responses = re.findall(r'Master Chef: (.*?)\n', conversation)
    trainee_responses = re.findall(r'Trainee Chef: (.*?)\n', conversation)
    return chef_responses, trainee_responses

def calculate_vocabulary_richness(text):
    words = text.split()
    return len(set(words)) / len(words)

def calculate_average_response_length(responses):
    return sum(len(response.split()) for response in responses) / len(responses)

def extract_culinary_terms(text, culinary_terms):
    return [term for term in culinary_terms if term.lower() in text.lower()]

def calculate_task_completion(responses, essential_steps):
    completed_steps = sum(1 for step in essential_steps if any(step.lower() in response.lower() for response in responses))
    return completed_steps / len(essential_steps)

def calculate_response_relevance(questions, answers):
    vectorizer = CountVectorizer().fit(questions + answers)
    question_vectors = vectorizer.transform(questions)
    answer_vectors = vectorizer.transform(answers)
    similarities = cosine_similarity(question_vectors, answer_vectors)
    return np.mean(np.diag(similarities))

def calculate_bleu_score(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def calculate_rouge_scores(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }


# Main evaluation pipeline
def evaluate_conversation(file_path, culinary_terms, essential_steps, reference_instructions):
    conversation = read_conversation(file_path)
    chef_responses, trainee_responses = separate_responses(conversation)
    
    chef_text = ' '.join(chef_responses)
    
    vocabulary_richness = calculate_vocabulary_richness(chef_text)
    avg_response_length = calculate_average_response_length(chef_responses)
    used_culinary_terms = extract_culinary_terms(chef_text, culinary_terms)
    task_completion = calculate_task_completion(chef_responses, essential_steps)
    response_relevance = calculate_response_relevance(trainee_responses, chef_responses)
    
    # Calculate BLEU and ROUGE scores
    bleu_score = calculate_bleu_score(reference_instructions, chef_text)
    rouge_scores = calculate_rouge_scores(reference_instructions, chef_text)
    
    return {
        'vocabulary_richness': vocabulary_richness,
        'average_response_length': avg_response_length,
        'culinary_terms_used': len(used_culinary_terms),
        'task_completion_rate': task_completion,
        'response_relevance': response_relevance,
        'bleu_score': bleu_score,
        'rouge_scores': rouge_scores
    }


# Example usage
culinary_terms = ['sauté', 'blanch', 'roast', 'simmer', 'dice', 'julienne']
essential_steps = ['preheat oven', 'season chicken', 'prepare vegetables', 'cook chicken', 'make sauce']

reference_instructions = """
Pat chicken dry with paper towels, season all over with 2 tsp. salt, and tie legs together with kitchen twine. Let sit at room temperature 1 hour.
Meanwhile, halve squash and scoop out seeds. Run a vegetable peeler along ridges of squash halves to remove skin. Cut each half into ½"-thick wedges; arrange on a rimmed baking sheet.
Combine sage, rosemary, and 6 Tbsp. melted butter in a large bowl; pour half of mixture over squash on baking sheet. Sprinkle squash with allspice, red pepper flakes, and ½ tsp. salt and season with black pepper; toss to coat.
Add bread, apples, oil, and ¼ tsp. salt to remaining herb butter in bowl; season with black pepper and toss to combine. Set aside.
Place onion and vinegar in a small bowl; season with salt and toss to coat. Let sit, tossing occasionally, until ready to serve.
Place a rack in middle and lower third of oven; preheat to 425°F. Mix miso and 3 Tbsp. room-temperature butter in a small bowl until smooth. Pat chicken dry with paper towels, then rub or brush all over with miso butter. Place chicken in a large cast-iron skillet and roast on middle rack until an instant-read thermometer inserted into the thickest part of breast registers 155°F, 50–60 minutes. (Temperature will climb to 165°F while chicken rests.) Let chicken rest in skillet at least 5 minutes, then transfer to a plate; reserve skillet.
"""

results = evaluate_conversation('conversation_3.1.txt', culinary_terms, essential_steps, reference_instructions)
print(results)
