import json
import random
import argparse

animals = ["butterfly", "cat", "chiken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

sentence_templates = [
    "I go to walk with my {animal}",
    "My {animal} loves to play in the park",
    "The {animal} sleeps on the window",
    "Yesterday I saw a {animal} at the zoo",
    "My friend has a pet {animal}",
    "The {animal} is running in the field",
    "We have a {animal} at home",
    "The {animal} is eating its food",
    "A {animal} crossed the road",
    "I took a picture of a {animal}",
    "The {animal} is a beautiful animal",
    "My neighbor's {animal} is very noisy",
    "I read a book about a {animal}",
    "The {animal} is sleeping in the sun",
    "That {animal} is very friendly",
    "I am afraid of the {animal}",
    "The {animal} has beautiful fur",
    "We visited a farm and saw a {animal}",
    "The {animal} jumped over the fence",
    "I dreamed about a {animal} last night",
    "My sister wants to buy a {animal}",
    "The {animal} lives in the forest",
    "Scientists discovered a new species of {animal}",
    "The {animal} made a strange noise",
    "My grandmother used to have a {animal}",
    "The {animal} is an endangered species",
    "Look at that {animal} over there!",
    "The {animal} was running very fast",
    "Can you see the {animal} in the distance?",
    "The {animal} has sharp teeth"
]

def generate_dataset(num_examples, output_file):
    dataset = []
    
    for _ in range(num_examples):
        animal = random.choice(animals)
        template = random.choice(sentence_templates)
        text = template.format(animal=animal)
        
        example = {
            "text": text,
            "label": animal
        }
        dataset.append(example)
    
    with open(output_file + '.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a dataset for animal text classification")
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to generate')
    parser.add_argument('--output_file', type=str, default="data/ner/animal_dataset", help='Output file path without extension')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() 
    generate_dataset(args.num_examples, args.output_file)
