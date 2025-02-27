from ner.inference import predict_animals as predict_text
from image_classification.inference import predict_animal as predict_img_class
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="This script classifies animals in both images and text. It combines image classification and Named Entity Recognition (NER) models to detect animals from input data.")
    parser.add_argument('--cv_img_path', type=str, required=True, help='Path to the image to classify')
    parser.add_argument('--cv_model_path', type=str, default='models/animal_recognition_model.pth', help='Path to the trained model file')
    parser.add_argument('--cv_num_classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--cv_data_dir', type=str, default='data/animals', help='Directory of the dataset (with ImageFolder structure)')
    
    parser.add_argument('--nlp_text', type=str, required=True, help='Input text to predict animals from')
    parser.add_argument('--nlp_model_path', type=str, default='models/animal_ner_model', help='Path to the pre-trained model and tokenizer')
    parser.add_argument('--nlp_animals_file', type=str, default='models/animal_ner_model/animals.json', help='Path to the file with animal names (JSON format)')
    return parser.parse_args()

# return true if there is a match between NLP and CV
def result(cv_detected_class, nlp_detected_class):
    for i in cv_detected_class:
        for j in nlp_detected_class:
            if i == j:
                return True
    return False

if __name__ == "__main__":
    args = parse_args() 
    
    cv_detected_class = predict_img_class(args.cv_img_path, args.cv_model_path, args.cv_num_classes, args.cv_data_dir)
    nlp_detected_class = predict_text(args.nlp_text, args.nlp_model_path, args.nlp_model_path, args.nlp_animals_file)

    ans = result(cv_detected_class, nlp_detected_class)
    
    print(f"CV: {cv_detected_class}")
    print(f"NLP: {nlp_detected_class}")
    
    print(ans)
