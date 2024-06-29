import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
classifier_dir = os.path.join(current_dir, 'classifier')
sys.path.append(classifier_dir)

from main import prediction


model = load_model('C:\\Users\\1mahe\\Desktop\\MessageClassifier\\msg_classifier_model.keras')

def classify_text():
    text = text_entry.get("1.0", tk.END).strip()
    if text:
        result = prediction(model, text)
        result_label.config(text=f"Prediction: {result}")
    else:
        messagebox.showwarning("Input Error", "Please enter some text to classify.")

# Initialize the main window
root = tk.Tk()
root.title("Message Classifier")

# Create a text entry widget
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=10)

# Create a button to classify the text
classify_button = tk.Button(root, text="Classify", command=classify_text)
classify_button.pack(pady=5)

# Create a label to display the result
result_label = tk.Label(root, text="Prediction: ")
result_label.pack(pady=5)

# Run the main loop
root.mainloop()
