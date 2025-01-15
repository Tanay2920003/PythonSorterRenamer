# Import libraries
import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import pytesseract

# Increase image size limit to handle large PDFs
Image.MAX_IMAGE_PIXELS = None

# Paths (update these as per your system)
training_folder = r"C:\Users\Legion\Desktop\Python\pdfsortrenamer\pdfsortrenamer\training_folder"
renamed_output_folder = r"C:\Users\Legion\Desktop\Python\pdfsortrenamer\pdfsortrenamer\renamed_output_folder"
unsorted_rename_folder = r"C:\Users\Legion\Desktop\Python\pdfsortrenamer\pdfsortrenamer\unsorted_rename_folder"

# Ensure the output folder exists
os.makedirs(renamed_output_folder, exist_ok=True)

# Function to extract text from PDFs using OCR
def extract_text_from_pdf(pdf_path):
    try:
        print(f"Extracting text from {pdf_path}...")
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
        extracted_text = ""
        for img in images:
            img = img.convert("L")  # Convert to grayscale
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)  # Increase contrast
            extracted_text += pytesseract.image_to_string(img)
        print(f"Text extraction from {pdf_path} complete.")
        return extracted_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function to prepare the dataset for training
def prepare_renaming_dataset(folder_path):
    print(f"Preparing dataset from folder: {folder_path}")
    data, labels = [], []
    for pdf_file in tqdm(os.listdir(folder_path), desc="Preparing dataset"):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                data.append(text)
                labels.append(os.path.splitext(pdf_file)[0])  # Use file name as label
    print(f"Dataset preparation complete. {len(data)} files processed.")
    return data, labels

# Prepare the dataset
texts, labels = prepare_renaming_dataset(training_folder)

if not texts or not labels:
    raise ValueError("No valid training data found. Check the dataset folder.")

# TF-IDF Vectorization
print("Performing TF-IDF vectorization...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)
print("TF-IDF vectorization complete.")

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
y = np.array([label_mapping[label] for label in labels])

# Train-test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# Train Logistic Regression model
print("Training Logistic Regression model...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Model training complete.")

# Model accuracy
print("Evaluating model accuracy...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Function to classify and rename PDF files
def classify_and_rename(pdf_path, model, vectorizer, output_folder):
    print(f"Classifying and renaming {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    if text.strip():
        X_input = vectorizer.transform([text])
        predicted_label_idx = model.predict(X_input)[0]
        predicted_label = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label_idx)]
        new_name = f"{predicted_label}.pdf"
        new_path = os.path.join(output_folder, new_name)
        shutil.move(pdf_path, new_path)  # Move and rename the file
        print(f"Renamed and moved: {pdf_path} -> {new_path}")
        return new_path
    else:
        print(f"Failed to classify {pdf_path}. No text extracted.")
        return None

# Process and rename files in the unsorted folder
def process_pdf_files():
    pdf_files = [f for f in os.listdir(unsorted_rename_folder) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files to process.")
    for pdf_file in tqdm(pdf_files, desc="Processing files"):
        pdf_path = os.path.join(unsorted_rename_folder, pdf_file)
        result = classify_and_rename(pdf_path, clf, vectorizer, renamed_output_folder)
        if result:
            print(f"Renamed and moved: {result}")
        else:
            print(f"Failed to process {pdf_file}")

# Start processing files
print("Starting to process files...")
process_pdf_files()
print("File processing complete.")
