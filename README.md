# Drug Classification ML CI/CD Project

A machine-learning project that predicts a drug category from patient attributes using a Random Forest classifier.

## Dataset

Place the Drug200 CSV file at `data/drug200.csv`.

Required columns: Age, Sex, BP, Cholesterol, Na_to_K, Drug.

## Run locally

Install dependencies:

    python -m pip install -r requirements.txt

Train and evaluate:

    python train.py

Outputs:
- `model/drug_model.joblib`
- `model/drug_model.onnx`
- `results/metrics.txt`
- `results/confusion_matrix.png`

## Web application

The static browser application is in `app/index.html`. It uses ONNX Runtime Web to load the exported model.

## Automation

- GitHub Actions CI trains and evaluates the model.
- GitHub Actions CD deploys the static app and ONNX model to Hugging Face Spaces.

Configure repository secrets `HF_TOKEN` and `HF_SPACE_ID` before deploying.

## Disclaimer

Educational demonstration only. Model predictions are not medical advice and must not be used to choose medication.
