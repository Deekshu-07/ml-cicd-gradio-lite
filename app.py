from pathlib import Path
import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "drug_classifier.joblib"
model = joblib.load(MODEL_PATH)


def predict_drug(age, sex, bp, cholesterol, na_to_k):
    data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "BP": bp,
        "Cholesterol": cholesterol,
        "Na_to_K": na_to_k,
    }])

    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    confidence = max(probabilities) * 100

    return (
        f"### Predicted drug category: {prediction}\n\n"
        f"**Model confidence:** {confidence:.1f}%\n\n"
        "Educational demonstration only. "
        "Do not use this prediction to select medication."
    )


with gr.Blocks(title="Drug Classification Dashboard") as demo:
    gr.Markdown("# Drug Classification Dashboard")
    gr.Markdown("Drug200 machine learning demonstration.")

    with gr.Row():
        age = gr.Slider(15, 100, value=35, step=1, label="Age")
        sex = gr.Dropdown(["F", "M"], value="F", label="Sex")

    with gr.Row():
        bp = gr.Dropdown(
            ["LOW", "NORMAL", "HIGH"],
            value="NORMAL",
            label="Blood Pressure"
        )
        cholesterol = gr.Dropdown(
            ["NORMAL", "HIGH"],
            value="NORMAL",
            label="Cholesterol"
        )

    na_to_k = gr.Slider(
        1, 40, value=15, step=0.1, label="Na_to_K Ratio"
    )

    predict_button = gr.Button(
        "Predict Drug Category", variant="primary"
    )
    output = gr.Markdown()

    predict_button.click(
        fn=predict_drug,
        inputs=[age, sex, bp, cholesterol, na_to_k],
        outputs=output
    )


if __name__ == "__main__":
    demo.launch()
