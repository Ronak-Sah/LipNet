# LipNet: Lip Reading using CNN + Transformer

This project implements a **Lip Reading (LipNet)** system that predicts spoken words directly from silent video input.  
The model uses a CNN-based visual front-end followed by an **encoder-only Transformer** for sequence modeling, optimized for both accuracy and low-latency inference.

---

## 🚀 Project Overview

- **Task**: Visual Speech Recognition (Lip Reading)
- **Input**: Silent video of lip movements
- **Output**: Predicted text sequence

---

## ✨ Features Implemented

- 3-layer **CNN** for visual feature extraction  
- **Encoder-only Transformer** for temporal modeling  
- Trained on **1000 video samples**  
- Achieved **Word Error Rate (WER) ≈ 19%**  
- Inference **API** using FastAPI  
- **TorchScript** conversion for faster inference  
- **~1.5× latency reduction** using TorchScript  
- Fully **Dockerized** application  

---

## 📊 Results

Final training evaluation:

Batch no : 1, Total batch : 7, Remaining batch : 6
Batch no : 2, Total batch : 7, Remaining batch : 5
Batch no : 3, Total batch : 7, Remaining batch : 4
Batch no : 4, Total batch : 7, Remaining batch : 3
Batch no : 5, Total batch : 7, Remaining batch : 2
Batch no : 6, Total batch : 7, Remaining batch : 1
Batch no : 7, Total batch : 7, Remaining batch : 0

Word error rate is : 0.1959523809523809

yaml
Copy code

➡️ **WER ≈ 19%**

---

## 🧠 Model Architecture

Input Video Frames
↓
3 × CNN Layers
↓
Encoder-only Transformer
↓
CTC Decoder
↓
Predicted Text

yaml
Copy code

---

## 🏋️ Training the Model

To train the LipNet model, run:

```bash
python main.py
```
## 🌐 Running the API
The trained model is served using FastAPI.

Start the API server with:

```bash
Copy code
uvicorn app:app --reload
```
The API accepts video input and returns the predicted text output and time taken for prediction .

## ⚡ TorchScript Optimization
The trained PyTorch model is converted to TorchScript

Benefits:

Reduced Python overhead

Faster inference

Deployment-ready model

Achieved ~1.5× inference speedup

## 🐳 Docker Support
The complete project is Dockerized for easy deployment.

Build Docker Image
bash
Copy code
docker build -t lipnet .
Run Docker Container
```bash
Copy code
docker run -p 8000:8000 lipnet
```
## 📌 Technologies Used
Python

PyTorch

Transformer (Encoder-only)

FastAPI

TorchScript

Docker

Uvicorn

## 🔮 Future Improvements
Train on larger-scale datasets

Use pretrained visual backbones

Integrate a language model for decoding

Apply quantization for further latency reduction

Real-time video inference support