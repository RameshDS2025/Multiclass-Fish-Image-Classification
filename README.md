# 🐟 Fish AI: Aquatic Species Intelligence

A premium web application for real-time multiclass fish image classification. Powered by Deep Learning (TensorFlow/Keras) and wrapped in a stunning, high-performance Streamlit interface with glassmorphism design.

## 🚀 Key Features
- **Instant AI Analysis**: Classify 9 different fish species with over 94% accuracy.
- **Micro-Animation Interface**: Fluid, responsive UI with themed gradients and bounce effects.
- **Glassmorphism Design**: Modern translucent components with premium backdrop blurs.
- **Species Catalog**: A high-resolution guide for exploring the dataset species.
- **Model Monitoring**: Real-time performance metrics and inference time tracking.

## 🧬 Supported Species
Our model is trained to identify:
- Black Sea Sprat
- Gilt Head Bream
- Hourse Mackerel
- Red Mullet
- Red Sea Bream
- Sea Bass
- Shrimp
- Striped Red Mullet
- Trout

## 🛠️ Project Structure
```text
Multiclass-Fish-Image-Classification/
├── streamlit_app/          # Core Streamlit application
│   └── app.py              # Main UI and Logic
├── models/                 # Pre-trained Keras models
│   └── BEST_FISH_MODEL.keras
├── notebooks/              # Research and Training notebooks
├── src/                    # Supporting source code
├── data/                   # Dataset structure (val/train/test)
├── fish_venv/              # Local virtual environment
├── requirements.txt        # Production dependencies
└── README.md
```

## 💻 Local Setup

### 1. Prerequisite
Ensure you have Python 3.9+ installed and Git configured.

### 2. Clone and Initialize
```bash
git clone https://github.com/RameshDS2025/Multiclass-Fish-Image-Classification.git
cd Multiclass-Fish-Image-Classification
```

### 3. Setup Environment
```bash
# Create virtual environment
python -m venv fish_venv

# Activate (Windows)
.\fish_venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run streamlit_app/app.py
```

## 📊 Model Specifications
- **Architecture**: Deep CNN (Transfer Learning ready)
- **Input Shape**: 224x224 RGB
- **Accuracy**: 94.2%
- **Inference Latency**: ~42ms

---
**Developed by RameshDS2025**  
*Empowering Marine Research via Artificial Intelligence*
