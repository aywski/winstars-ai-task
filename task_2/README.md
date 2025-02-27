# Named Entity Recognition and Animal Classification Pipeline

## Overview
This project implements a pipeline that takes an image of an animal and a text description and decides if the animal described in the text is present in the image. The project consists of two main models:
1. **Named Entity Recognition (NER)** model for extracting animal names from text.
    - A transformer-based NER model is trained to extract animal names from text descriptions.
    - The model is fine-tuned on a [custom dataset](https://github.com/aywski/winstars-ai-task/blob/main/task_2/src/ner/dataset_generator.py) with animal-related entities.

2. **Animal Classification** model for detecting animals in images.
    - An image classification model is trained to classify animals in images.
    - The model is based on ResNet18 and is trained on the [Animal Image Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data) (10 Different Animals) from Kaggle.

## Steps to Run the pretrained Pipeline:
1. Clone the repository.
```bash
git clone https://github.com/aywski/winstars-ai-task
```
2. Create a virtual environment: 
```bash
cd task2
python -m venv .venv
.venv\Scripts\activate    # on Windows 
source .venv/bin/activate   # on Linux
```
3. Install dependencies: `pip install -r requirements.txt`.
4. Use the `pipeline.py` script to decide whether the text and image match:
```bash
#example

python ./src/pipeline.py --cv_img_path "image.jpg" --nlp_text "the dog will walk while my cat sleeps"
```

## Repository Structure
```
task_2/
│
├── data/
│   ├── animals/
│   └── ner/
│
├── models/
│   ├── animal_ner_model/
│   └── animal_recognition_model.pth
│
├── src/
│   ├── image_classification/
│   │   └── inference.py
│   ├── ner/
│   │   ├── dataset_generator.py
│   │   ├── inference.py
│   │   ├── train.py
│   │   └── pipeline.py
│
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Scripts Overview
#### 1. `image_classification/train.py`
- Trains an ResNet18 image classification model on the animal dataset.
- Saves the trained model to models/animal_recognition_model.pth (by default).
- Saves the trained model to models/animal_ner_model/ (by default).
#### 2. `image_classification/inference.py`
- Loads the trained Image Classification model and predict the animal in user-provided image.
#### 3. `ner/train.py`
- Trains a transformer-based NER model to extract 10 animal names from text.
- Saves the trained model to models/animal_ner_model/ (by default).
#### 4. `ner/inference.py`
- Loads the trained NER model and extracts animal names from user-provided text.
#### 5. `ner/dataset_generator.py`
- Generate custom NER Dataset 
#### 6. `pipeline.py`
- Integrates the NER and Image Classification models.
- Takes a text description and an image as inputs, and outputs True if the animal in the text matches the animal in the image, else False.

## Requirements
- __PyTorch, Transformers__
- numpy, scikit-learn, matplotlib, PIL, OpenCV
- Python 3.12.4

## Pipeline Arguments:
```ps
python ./src/pipeline.py --cv_img_path CV_IMG_PATH [--cv_model_path CV_MODEL_PATH] [--cv_num_classes CV_NUM_CLASSES] [--cv_data_dir CV_DATA_DIR] --nlp_text NLP_TEXT [--nlp_model_path NLP_MODEL_PATH] [--nlp_animals_file NLP_ANIMALS_FILE]
```

## Developed By
__Sahalianov Arsenii__ for Winstars AI 