# SmartCrop AI – Smart Crop Recommendation System

A Flask + Scikit-learn academic project that recommends the top three crop classes from seven soil and climate inputs.

## Features
- Crop classification using a model selected after comparing Random Forest, Decision Tree, KNN and Gaussian Naive Bayes.
- Top-three model class scores.
- User registration, login and saved recommendation history (SQLite).
- Responsive Bootstrap interface.
- Training metrics and confusion-matrix output.
- Input validation and basic automated route tests.

## Requirements
- Python 3.10–3.12 recommended
- VS Code
- Internet for initial dependency and dataset download
- Windows, macOS or Linux

## Windows setup (PowerShell)
```powershell
cd SmartCrop-AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python ml/fetch_data.py
python ml/train_model.py
python app.py
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```
Open http://127.0.0.1:5000 in your browser.

## macOS / Linux setup
```bash
cd SmartCrop-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python ml/fetch_data.py
python ml/train_model.py
python app.py
```

## First test
Use the following example values to check that the form submits (this is an example input, not a guarantee of field suitability):
- N: 90
- P: 42
- K: 43
- Temperature: 20.9
- Humidity: 82
- pH: 6.5
- Rainfall: 203

## Tests
After installing dependencies and training the model:
```bash
python -m unittest discover -s tests -v
```

## Project files
- `ml/fetch_data.py`: downloads the public CSV dataset.
- `ml/train_model.py`: cleans, splits, trains, compares and evaluates candidate classifiers.
- `ml/predict.py`: loads the saved model and ranks its class scores.
- `app.py`: Flask routes and application logic.
- `database/db.py`: SQLite data models.
- `models/`: generated model and evaluation metadata.
- `static/images/confusion_matrix.png`: generated confusion matrix.

## Dataset and academic attribution
The project uses the Crop Recommendation CSV hosted by the Hugging Face `liaad/agricultural-data` dataset repository:
https://huggingface.co/datasets/liaad/agricultural-data

The commonly used source dataset contains seven input attributes (N, P, K, temperature, humidity, pH, rainfall) and a crop label. Check the source repository's current licence and cite the dataset in your final report. Do not describe this dataset as a representative survey of every Indian farm.

## Important interpretation note
The top-three percentages are classifier scores derived from `predict_proba`; they are not calibrated probabilities of yield, profit, or successful harvest. The training dataset is an educational benchmark and has limitations in geographic, seasonal and agronomic coverage. Validate real farming decisions with local agricultural experts and soil-test reports.

## Before public deployment
Change `SECRET_KEY`, disable Flask debug mode, add CSRF protection, configure secure cookies and HTTPS, add rate limiting, review privacy requirements and test authentication thoroughly. This starter project is intended for local academic demonstration, not production deployment.
