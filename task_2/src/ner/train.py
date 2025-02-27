import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import re
import argparse

# creating class for dataset
class AnimalDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        all_animals = set()
        for example in self.data:
            all_animals.add(example['label'])
        
        self.animals = list(all_animals)
        self.animal_to_id = {animal: idx for idx, animal in enumerate(self.animals)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        text = example['text']
        animal = example['label']
        
        # finding position of animal in text
        positions = [m.start() for m in re.finditer(animal, text.lower())]
        
        # text tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # creating label for each token
        labels = torch.zeros(encoding['input_ids'].size(1), dtype=torch.long)
        
        # find tokens relevant to the animal
        offset_mapping = encoding['offset_mapping'][0]
        
        for start_pos in positions:
            end_pos = start_pos + len(animal)
            
            # find tokens that cover the animal
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start.item() <= start_pos and token_end.item() >= start_pos:
                    # B-ANIMAL (beggining of entity)
                    labels[i] = 1
                elif token_start.item() > start_pos and token_end.item() <= end_pos:
                    # I-ANIMAL (continuation of entity)
                    labels[i] = 2
        
        encoding.pop('offset_mapping')
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = labels
        
        return item

def load_and_split_data(file_path, train_ratio=0.8, val_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio - 1.0) < 1e-5, "The ratios should add up to 1"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.seed(seed)
    random.shuffle(data)
    
    dataset_size = len(data)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    
    print(f"Splitted data: {len(train_data)} for training, {len(val_data)} for validation")
    
    return train_data, val_data


def train_model(model, train_dataloader, val_dataloader, epochs=3, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}\n\n')
        
        # start training
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f'Training loss: {avg_train_loss}')
        
        # validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch) # pass the data through the model
                loss = outputs.loss
                
                val_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
                
                active_labels = batch['labels'].view(-1) != -100
                if active_labels.sum() > 0:
                    val_preds.extend(predictions.view(-1)[active_labels].cpu().numpy())
                    val_true.extend(batch['labels'].view(-1)[active_labels].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Validation loss: {avg_val_loss}')
        
        if len(val_preds) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='weighted')
            accuracy = accuracy_score(val_true, val_preds)
            
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1-score: {f1:.4f}')
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for animal NER task")
    parser.add_argument('--data_file', type=str, default='data/ner/animal_dataset.json', help='Path to the data file in JSON format')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--model_save_path', type=str, default='models/animal_ner_model', help='Directory to save the trained model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() 
    
    print("Loading and splitting data...")
    train_data, val_data = load_and_split_data(args.data_file, train_ratio=0.8, val_ratio=0.2)
    
    tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
    model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
    
    train_dataset = AnimalDataset(train_data, tokenizer)
    val_dataset = AnimalDataset(val_data, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    model = train_model(model, train_dataloader, val_dataloader, epochs=args.epochs, lr=args.lr)
    
    # save model
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    
    # save list of animals
    all_animals = train_dataset.animals
    with open(f'{args.model_save_path}/animals.json', 'w', encoding='utf-8') as f:
        json.dump(all_animals, f, indent=2)
    
    print(f"The model is saved in the {args.model_save_path} directory.")
