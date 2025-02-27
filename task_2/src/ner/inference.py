import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import argparse

def predict_animals(text, model, tokenizer, animals):
    model = AutoModelForTokenClassification.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    with open(animals, 'r', encoding='utf-8') as f:
        animals = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # text tokenization
    inputs = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    # getting token offsets
    offset_mapping = inputs.pop('offset_mapping')
    offset_mapping = offset_mapping.numpy()[0]
    
    # transfer input data to the selected device (GPU/CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    predictions = predictions[0].cpu().numpy()
    
    animal_tokens = []
    
    # process all tokens to collect all animal references
    for i, pred in enumerate(predictions):
        if pred == 1:  # B-ANIMAL (beggining of entity)
            start_idx = i
            end_idx = i
            for j in range(i + 1, len(predictions)):
                if predictions[j] == 2:  # I-ANIMAL (continuation of entity)
                    end_idx = j
                else:
                    break
            
            if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
                start_pos = offset_mapping[start_idx][0]
                end_pos = offset_mapping[end_idx][1]
                
                if start_pos < len(text) and end_pos <= len(text):
                    detected_animal = text[start_pos:end_pos].lower()
                    animal_tokens.append(detected_animal)
    
    detected_animals = []
    if animal_tokens:
        for detected in animal_tokens:
            best_match = None
            best_score = 0
            for animal in animals:
                if animal in detected or detected in animal:
                    score = len(set(animal) & set(detected))
                    if score > best_score:
                        best_score = score
                        best_match = animal
            detected_animals.append(best_match if best_match else detected)
    
    # return all found animal references
    return detected_animals if detected_animals else ["No animals detected"]

def parse_args():
    parser = argparse.ArgumentParser(description="Predict animals from text using a token classification model.")
    parser.add_argument('--model_path', type=str, default='models/animal_ner_model', help='Path to the pre-trained model and tokenizer')
    parser.add_argument('--animals_file', type=str, default='models/animal_ner_model/animals.json', help='Path to the file with animal names (JSON format)')
    parser.add_argument('--text', type=str, required=True, help='Input text to predict animals from')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() 

    detected_animals = predict_animals(args.text, args.model_path, args.model_path, args.animals_file)
    print(f"Input Text: {args.text}")
    print(f"Detected animals: {detected_animals}")
