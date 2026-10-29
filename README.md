
# Phishing Website Detection Using Machine Learning

**Program:** IBM SkillsBuild Cyber Security BootCamp 2026  
**Conducted by:** BharatCares  
**Project Author:** Sumit Raj  
**Branch:** B.Tech (EE)  
**Institution:** Budge Budge Institute of Technology

## Project Overview

This project uses machine learning to classify records from a phishing-related website dataset. A Random Forest classifier is trained and evaluated to demonstrate how machine learning can support cybersecurity analysis.

## Objectives

- Study phishing website classification.
- Load and prepare a public dataset.
- Train a Random Forest model.
- Evaluate model performance using accuracy, precision, recall and F1-score.
- Generate a confusion matrix and feature-importance chart.

## Dataset

**Dataset:** Phishing Websites  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 327

Official link:
https://archive.ics.uci.edu/dataset/327/phishing+websites

## Technologies Used

- Python
- pandas
- scikit-learn
- Matplotlib
- UCI Machine Learning Repository
- Visual Studio Code

## Project Files

- `SumitRaj_PhishingURLDetection.py` — Main Python source code
- `requirements.txt` — Required Python libraries
- `SumitRaj_ProjectReport.docx` — Project report
- `README.md` — Project documentation

## Installation

Install Python 3.10 or later.

Open the terminal in VS Code and run:

```bash
python -m venv .venv
```

Activate the virtual environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

```bash
python SumitRaj_PhishingURLDetection.py
```

An internet connection is required to retrieve the dataset.

## Outputs

The program generates an `outputs` folder containing:

- `evaluation_results.txt`
- `confusion_matrix.png`
- `feature_importance.png`
- `feature_importance.csv`
- `test_predictions.csv`
- `target_label_mapping.txt`

## Limitations

This project is a dataset-based classification demonstration. It is not a live URL scanner and cannot guarantee that a website is safe.

## Conclusion

This project demonstrates a basic machine-learning workflow for a cybersecurity classification task, including data preparation, model training, evaluation and visualization.
