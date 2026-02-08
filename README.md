# WaferGuard-Intelligence

Project Title: Edge-AI Based Wafer Defect Classification using Fine-Tuned MobileNetV2

Problem Statement: Semiconductor manufacturing generates large volumes of wafer inspection images. Traditional centralized or manual inspection systems suffer from high latency, bandwidth limitations, and scalability challenges. The objective is to design a lightweight, accurate, and edge-deployable AI system capable of real-time wafer defect classification under compute constraints and compatible with ONNX-based deployment.

Architecture

Model: MobileNetV2

Approach: Transfer Learning + Fine-Tuning

Input Processing: Grayscale conversion → Resize to 224×224

Classes: 9 (7 Defects + Clean + Other)

Deployment Format: ONNX

Edge-Compatible: Yes (NXP eIQ compatible)

Model Performance (Test Split)

Test Accuracy: 95.73%

Best Validation Accuracy: 96.01%

Macro Precision: 0.96

Macro Recall: 0.96

Macro F1-Score: 0.96

Total Test Samples: 702 (Balanced)

Model Size

PyTorch Model (.pth): 8.75 MB

ONNX Model (model.onnx + model.onnx.data): ~9 MB

How to Run the Code

1. Clone Repository

git clone https://github.com/yourusername/EdgeAI-Wafer-Defect-Classification.git
cd EdgeAI-Wafer-Defect-Classification

2. Create Virtual Environment

python -m venv venv

venv\Scripts\activate

3. Install Dependencies

pip install torch torchvision scikit-learn matplotlib seaborn onnx onnxscript

4. Place Dataset in Below Order

dataset/

   train/
   
   valid/
   
   test/

5. Train Model
python train_model.py

6. Export ONNX
python export_onnx.py


Dataset Link

Google Drive Dataset ZIP: https://drive.google.com/file/d/1KiYrbPARF8RMqgSeZ0gBdamA3KTnM_uB/view?usp=drive_link

ONNX Model Link

ONNX Model (model.onnx + model.onnx.data): https://drive.google.com/drive/folders/1TSRfi36jtpNPHMrXe9eGl23A70VGE8S5?usp=drive_link
