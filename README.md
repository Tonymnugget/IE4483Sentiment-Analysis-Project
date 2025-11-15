# IE4483-Sentiment-Analysis-Project
Two sentiment analysis pipelines using a Gemma-based transformer model and a TF-IDF + Feed-Forward Neural Network (MLP). Includes full preprocessing, hyperparameter tuning (Optuna), evaluation scripts, and reproducible training code.

## gemma_pipeline

* Custom dataset preprocessing (data_gemma.py)
* Model definiton (model_gemma.py)
* Evaluation utilities (eval_gemma.py)
* Custom trainer class (train_gemma.py)
* Training start point (main_gemma.py)
* Output directory for metrics and plots (gemma_eval folder)
* Model Prediction results on test.json (submission_gemma.csv)
* Labeled training reviews (train.json) 
* Unlabeled test reviews (test.json)

## tfidf_pipeline

* Custom dataset preprocessing (data.py)
* Custom TF-IDF script (features_tfidf.py)
* Model definition (models_tfidf.py)
* Evalution utilities (eval_tfidf.py)
* Custom trainer class (train_tfidf.py)
* Training start point (main_tfidf.py)
* Package requirements for Venv (requirements.txt)
* Output directory for metrics and plots (tfidf_eval folder)
* Model Prediction results on test.json (submission.csv)
* Labeled training reviews (train.json) 
* Unlabeled test reviews (test.json)

## How to run

1. Install dependencies using requirements.txt in tfidf_pipeline
   ```bash
   pip install -r requirements.txt
   ```
   Recommended to create virtual environment via vscode or miniconda then install dependencies there

2. Running the tfidf_pipeline
   ```bash
   python main_tfidf.py
   ```
3. Running the gemma_pipeline
   ```bash
   python main_gemma.py
   ```
