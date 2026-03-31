# 🧠 Image Segmentation using Traditional Computer Vision

## 📌 Overview

This project focuses on segmenting objects from images using traditional computer vision techniques without relying on deep learning models. The goal is to extract meaningful regions from images using methods like thresholding and edge detection.

---

## 🎯 Problem Statement

Image segmentation is an essential task in computer vision used to identify and isolate objects within an image. This project aims to implement segmentation using classical OpenCV-based approaches.

---

## 🚀 Features

* Image preprocessing
* Edge detection using Canny
* Threshold-based segmentation
* Contour detection and extraction
* Visualization of segmented output

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib

---

## 📂 Project Structure

```
image-segmentation-cv-project/
│
├── app.py
├── segmentation.py
├── images/
├── output/
├── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/image-segmentation-cv-project.git
```

2. Navigate to the folder:

```
cd image-segmentation-cv-project
```

3. Install dependencies:

```
pip install opencv-python numpy matplotlib
```

4. Run the project:

```
python app.py
```

---

## 📷 Output

* Original Image
* Edge Detected Image
* Segmented Output

---

## ⚙️ Methodology

1. Load the input image
2. Convert to grayscale
3. Apply Gaussian blur
4. Perform edge detection (Canny)
5. Apply thresholding
6. Detect contours
7. Extract segmented regions

---

## ⚠️ Challenges

* Sensitivity to lighting conditions
* Noise in images
* Choosing optimal threshold values

---

## 📈 Future Scope

* Integration with deep learning models like U-Net
* Real-time segmentation using webcam
* Improved accuracy with adaptive thresholding

---

## 📚 Learnings

* Practical understanding of image segmentation
* Hands-on experience with OpenCV
* Importance of preprocessing in CV tasks

---

## 👨‍💻 Author

Prateek
