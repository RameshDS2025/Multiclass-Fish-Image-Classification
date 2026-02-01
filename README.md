# Multiclass Fish Image Classification

This project focuses on classifying fish images into multiple categories using deep learning models.

## Structure
- `data/`: Dataset folder
- `src/`: Source code for training and evaluation
- `app/`: Streamlit web application
- `models/`: Trained models
- `notebooks/`: Jupyter notebooks for experiments


## Data Setup
The code expects the following directory structure inside `data/`:
```
data/
    train/
        class_1/
        class_2/
        ...
    val/
        class_1/
        ...
    test/
        class_1/
        ...
```
Please copy your images from `Dataset` into these folders or update the path in `src/data_loader.py` (or the notebook).

## Usage
### Running in Colab
1.  Upload the entire project folder to Google Drive.
2.  Open `notebooks/Model_Training_Colab.ipynb` in Colab.
3.  Follow the instructions in the notebook to mount Drive and start training.

### Local Execution (If applicable)
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run Streamlit app: `streamlit run app/app.py`
