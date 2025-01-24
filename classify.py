import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class PetClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pet Classifier")
        self.root.geometry("600x600")
        self.root.configure(bg='#f0f0f0')

        try:
            print("Attempting to load model...")
            self.model = tf.keras.models.load_model("dog_vs_cat_classifier.h5")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}\nMake sure dog_vs_cat_classifier.h5 exists in the current directory.")
            self.root.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        
        title_label = tk.Label(
            self.root,
            text="Dog vs Cat Classifier",
            font=("Arial", 20, "bold"),
            bg='#f0f0f0'
        )
        title_label.pack(pady=10)

  
        self.image_frame = tk.Frame(self.root, bg='white', width=300, height=300, relief='solid', borderwidth=1)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg='white')
        self.image_label.pack(expand=True, fill='both')

        
        self.browse_button = tk.Button(
            self.root, text="Browse Image", command=self.browse_image, font=("Arial", 12), width=15
        )
        self.browse_button.pack(pady=10)

        self.classify_button = tk.Button(
            self.root, text="Classify", command=self.classify_image, font=("Arial", 12), width=15, state=tk.DISABLED
        )
        self.classify_button.pack(pady=10)

        self.result_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.result_frame.pack(pady=20)

        self.result_label = tk.Label(
            self.result_frame,
            text="No prediction yet",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='black'
        )
        self.result_label.pack()

        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='black'
        )
        self.confidence_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if file_path:
            print(f"Selected image: {file_path}")
            try:
                image = Image.open(file_path)
                display_size = (280, 280)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)

                self.image_label.configure(image=photo)
                self.image_label.image = photo  
                self.current_image_path = file_path
                self.classify_button.configure(state=tk.NORMAL)

                self.result_label.configure(text="Ready to classify")
                self.confidence_label.configure(text="")
            except Exception as e:
                print(f"Error opening image: {str(e)}")
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")

    def classify_image(self):
        try:
            print("Starting classification...")
            img = load_img(self.current_image_path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = self.model.predict(img_array, verbose=0)[0][0]
            is_dog = prediction > 0.5
            confidence = prediction if is_dog else 1 - prediction

            result_text = "Dog" if is_dog else "Cat"
            confidence_text = f"Confidence: {confidence * 100:.2f}%"

            self.result_label.configure(text=f"Prediction: {result_text}", fg='green' if confidence > 0.75 else 'orange')
            self.confidence_label.configure(text=confidence_text)

            print(f"Result: {result_text}, Confidence: {confidence * 100:.2f}%")
        except Exception as e:
            print(f"Error classifying image: {str(e)}")
            messagebox.showerror("Error", f"Failed to classify image: {str(e)}")

def main():
    root = tk.Tk()
    app = PetClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


