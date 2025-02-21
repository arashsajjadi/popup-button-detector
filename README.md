# 🚀 Pop-up Button Detector (AI-Powered)

## **📌 Introduction**
This project provides an **AI-powered system to automatically detect and locate pop-up buttons** in UI screenshots (such as "Not Now", "OK", "Cancel", etc.). It combines **YOLO for object detection, EasyOCR for text extraction, and LLaVA for text-based decision-making**.

The model is designed to work with **mobile UI screenshots** and can be fine-tuned to detect different types of buttons based on a dataset.

---

## **⚙️ Features**
- **YOLO Object Detection** 🖼️: Identifies UI elements like buttons using a pre-trained YOLO model.
- **EasyOCR Text Recognition** 🔠: Extracts text from images to identify button labels.
- **LLaVA Language Model** 🧠: Determines which button corresponds to the "dismiss" action.
- **Auto-Clicking Logic** 🖱️: Computes the exact **click coordinates** of the target button.
- **Customizable and Fine-Tunable** 🔧: YOLO can be fine-tuned for better detection performance.

---

## **📂 Project Structure**
```
📦 popup-button-detector
│── 📄 popup_button_detector.ipynb  # Jupyter Notebook with complete code
│── 📄 README.md                    # Project documentation
│── 📄 requirements.txt              # List of dependencies
│── 📂 dataset/                      # Folder containing images and labels for training
│── 📂 models/                        # Pre-trained and fine-tuned YOLO models
│── 📂 results/                       # Output images with detected buttons
│── 📂 scripts/                       # Python scripts for training and inference
│── 📂 examples/                      # Sample test images
```

---

## **🛠️ Setup & Installation**
### **1️⃣ Install Dependencies**
Make sure you have Python 3.8+ installed. Then, install the required packages using:

```bash
pip install -r requirements.txt
```

Alternatively, install the core dependencies manually:

```bash
pip install ultralytics easyocr opencv-python matplotlib numpy subprocess
```

> If you plan to fine-tune YOLO, **GPU acceleration is recommended** (NVIDIA CUDA for PyTorch).

---

## **📜 How It Works**
### **🔹 Step 1: Load an Image**
The script takes an image as input and preprocesses it:

```python
orig_image, rgb_image = preprocess_image("example.jpg")
```

### **🔹 Step 2: Detect UI Buttons Using YOLO**
The **YOLO model** detects UI elements (such as buttons):

```python
yolo_detections = detect_buttons_yolo(rgb_image)
```
If no objects are detected, it prints:

```
⚠ YOLO Raw Detections: 0
✅ YOLO Filtered Detections: 0
```

To improve detection:
- Lower the confidence threshold in `detect_buttons_yolo()`.
- Fine-tune YOLO (explained below).

### **🔹 Step 3: Extract Text Using EasyOCR**
The **EasyOCR model** extracts all readable text from the image:

```python
ocr_results = extract_text_easyocr(rgb_image)
```

Example output:

```
OCR Extracted Text: ["Not Now", "Cancel", "OK", "Settings"]
```

### **🔹 Step 4: Ask LLaVA to Identify the Close Button**
The system **queries a language model (LLaVA)** to determine the correct button to click:

```python
chosen_text = query_llava_for_button(candidate_texts)
```

Example query:

```
We have extracted the following button texts from a pop-up:
["Not Now", "Cancel", "OK", "Settings"]
Which one is the button that closes or dismisses the pop-up?
```

Expected response:
```
LLaVA Selected: "Not Now"
```

### **🔹 Step 5: Find the Clickable Button**
Once LLaVA selects the button, the system **matches the selected text** with OCR results using fuzzy matching:

```python
selected_candidate = fuzzy_match_candidate(candidates, chosen_text)
```

If a match is found, it extracts the button **bounding box** and calculates the **click position**.

### **🔹 Step 6: Calculate the Click Coordinates**
The exact **center coordinates** of the detected button are computed:

```python
x1, y1, x2, y2 = selected_candidate['bbox']
click_x = (x1 + x2) / 2.0
click_y = (y1 + y2) / 2.0
norm_x = click_x / orig_image.shape[1]
norm_y = click_y / orig_image.shape[0]

print(f"🖱 Click Coordinates: ({click_x}, {click_y}) | Normalized: ({norm_x:.2f}, {norm_y:.2f})")
```

Example output:

```
🖱 Click Coordinates: (274, 975) | Normalized: (0.35, 0.57)
```

### **🔹 Step 7: Visualize the Detected Buttons**
The system draws bounding boxes on the image and saves the output.

```python
output_image = draw_all_detections(orig_image, ocr_results, yolo_detections, selected_candidate)
cv2.imwrite("output_N_example.jpg", output_image)
```

---

## **🛠️ Fine-Tuning YOLO for Better Accuracy**
If YOLO is **not detecting buttons correctly**, fine-tuning it on a **custom dataset** will improve results.

### **🔹 1. Prepare the Dataset**
Create a folder **`dataset/`** and add labeled images in YOLO format:
```
dataset/
│── images/
│   │── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │── val/
│   │   ├── image3.jpg
│   │   ├── image4.jpg
│── labels/
│   │── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │── val/
│   │   ├── image3.txt
│   │   ├── image4.txt
```

### **🔹 2. Create the Dataset YAML File**
Save this as `popups.yaml`:
```yaml
train: dataset/images/train
val: dataset/images/val
names:
  0: close_button
```

### **🔹 3. Train YOLO with Your Dataset**
```python
from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolo11x.pt')

# Train the model on custom dataset
model.train(data='popups.yaml', 
            epochs=50, 
            batch=16, 
            imgsz=640)

print("Training complete!")
```

### **🔹 4. Evaluate the Fine-Tuned Model**
```python
# Evaluate model performance
metrics = model.val(data='popups.yaml')
print(metrics)
```

### **🔹 5. Use Fine-Tuned Model for Inference**
```python
# Load the fine-tuned model
model = YOLO('runs/train/exp/weights/best.pt')

# Detect buttons in a new image
results = model("example.jpg")
results.show()
```

---

## **🔍 Future Improvements**
✅ Improve YOLO model with more data.  
✅ Use **LoRA** to fine-tune LLaVA to reduce mistakes.  
✅ Add **multi-modal reasoning** to improve decision accuracy.

---

## **📜 License**
This project is open-source under the **MIT License**.  

---

## **👨‍💻 Author**
**Developed by [Your Name]**  
Contact: [Your Email]  
GitHub: [Your GitHub Profile]  

---

## **🌟 Support & Contributions**
If you find this project useful, **please give it a ⭐ on GitHub**! Contributions are welcome.
