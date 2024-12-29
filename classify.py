
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os


class PetClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pet Classifier")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        # Load the trained model
        try:
            print("Attempting to load model...")
            self.model = tf.keras.models.load_model("dog_vs_cat_classifier.h5")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            messagebox.showerror(
                "Error",
                f"Failed to load model: {str(e)}\nMake sure dog_vs_cat_classifier.h5 exists in the current directory.",
            )
            self.root.destroy()
            return

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Dog vs Cat Classifier",
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
        )
        title_label.pack(pady=20)

        # Image display area
        self.image_frame = tk.Frame(
            self.root, width=300, height=300, bg="white", relief="solid", borderwidth=1
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(expand=True, fill="both")

        # Buttons frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=20)

        # Browse button
        self.browse_button = tk.Button(
            button_frame,
            text="Browse Image",
            command=self.browse_image,
            font=("Arial", 12),
            width=15,
        )
        self.browse_button.pack(side=tk.LEFT, padx=10)

        # Classify button
        self.classify_button = tk.Button(
            button_frame,
            text="Classify",
            command=self.classify_image,
            font=("Arial", 12),
            width=15,
            state=tk.DISABLED,
        )
        self.classify_button.pack(side=tk.LEFT, padx=10)

        # Result label
        self.result_label = tk.Label(
            self.root,
            text="No prediction yet",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
        )
        self.result_label.pack(pady=20)

    def browse_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                    ("All files", "*.*"),
                ]
            )

            if file_path:
                print(f"Selected image: {file_path}")
                # Open and display the image
                image = Image.open(file_path)
                display_size = (280, 280)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)

                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Keep a reference

                # Store the file path for classification
                self.current_image_path = file_path
                self.classify_button.configure(state=tk.NORMAL)

                # Clear previous results
                self.result_label.configure(text="Ready to classify")
                print("Image loaded successfully")

        except Exception as e:
            print(f"Error in browse_image: {str(e)}")
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")

    def classify_image(self):
        try:
            print("Starting classification...")
            # Load and preprocess the image
            img = load_img(self.current_image_path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            prediction = self.model.predict(img_array, verbose=1)[0][0]
            print(f"Raw prediction value: {prediction}")

            # Determine class and confidence
            is_dog = prediction > 0.5
            confidence = prediction if is_dog else 1 - prediction

            # Open a new result window
            self.show_result_window(is_dog, confidence)

        except Exception as e:
            print(f"Error in classify_image: {str(e)}")
            messagebox.showerror("Error", f"Failed to classify image: {str(e)}")

    def show_result_window(self, is_dog, confidence):
        # Create a new Toplevel window
        result_window = tk.Toplevel(self.root)
        result_window.title("Classification Result")
        result_window.geometry("400x300")
        result_window.configure(bg="#f0f0f0")

        # Display result message
        result_text = "It's a Dog!" if is_dog else "It's a Cat!"
        result_label = tk.Label(
            result_window,
            text=result_text,
            font=("Arial", 20, "bold"),
            fg="#006400" if is_dog else "#CC5500",
            bg="#f0f0f0",
        )
        result_label.pack(pady=20)

        # Optional: Display an icon (use your own dog/cat images)
        icon_path = "dog_icon.png" if is_dog else "cat_icon.png"
        if os.path.exists(icon_path):
            icon_image = Image.open(icon_path)
            icon_image.thumbnail((100, 100), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            icon_label = tk.Label(result_window, image=icon_photo, bg="#f0f0f0")
            icon_label.image = icon_photo  # Keep a reference
            icon_label.pack()

        # Display confidence
        confidence_label = tk.Label(
            result_window,
            text=f"Confidence: {confidence * 100:.2f}%",
            font=("Arial", 14),
            bg="#f0f0f0",
        )
        confidence_label.pack(pady=10)

        # Add a close button
        close_button = tk.Button(
            result_window, text="Close", command=result_window.destroy, font=("Arial", 12)
        )
        close_button.pack(pady=20)


def main():
    root = tk.Tk()
    app = PetClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
