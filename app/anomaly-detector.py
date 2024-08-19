import gradio as gr
import pickle
import pandas as pd

with open("./model/anomaly_detector_pipeline.pkl", "rb") as model_file:
    pipe = pickle.load(model_file)

def predict_class(timestamp, p_pdg, p_tpt, t_tpt, p_mon_ckp, t_jus_ckp, p_jus_ckgl, t_jus_ckgl, qgl):
    """Predict flow instability anomaly type of observation based on sensors measurements.
    Args:
        timestamp (str): Timestamp of the observation
        p_pdg (float): Pressure at the PDG (permanent downhole gauge) [Pa]
        p_tpt (float): Pressure at the TPT (temperature and pressure transducer) [Pa]
        t_tpt (float): Temperature at the TPT (temperature and pressure transducer) [ºC]
        p_mon_ckp (float): Upstream pressure of the PCK (production choke) [Pa]
        t_jus_ckp (float): Downstream temperature of the PCK (production choke) [ºC]
        p_jus_ckgl (float): Downstream pressure of the GLCK (gas lift choke) [Pa]
        t_jus_ckgl (float): Downstream temperature of the GLCK (gas lift choke) [ºC]
        qgl (float): Gas lift flow rate [m³/s]
    Returns:
        str: Predicted class
    """
    columns = ["timestamp", "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL", "T-JUS-CKGL", "QGL"]
    features = [timestamp, p_pdg, p_tpt, t_tpt, p_mon_ckp, t_jus_ckp, p_jus_ckgl, t_jus_ckgl, qgl]
    predicted_class = pipe.predict(pd.DataFrame([features], columns=columns))[0]
    if predicted_class==4.0:
        predicted_class="Anomaly"
    else:
        predicted_class="Normal"
    label = f"Predicted Class: {predicted_class}"
    return label


inputs = [
    gr.Textbox(label="Timestamp of the observation"),
    gr.Slider(-50000000, 50000000, step=1000, label="Pressure at the PDG [Pa]"),
    gr.Slider(0, 50000000, step=1000, label="Pressure at the TPT [Pa]"),
    gr.Slider(0, 200, step=1, label="Temperature at the TPT [ºC]"),
    gr.Slider(0, 50000000, step=1000, label="Upstream pressure of the PCK [Pa]"),
    gr.Slider(0, 200, step=1, label="Downstream temperature of the PCK [ºC]"),
    gr.Slider(0, 4000000000, step=100000, label="Downstream pressure of the GLCK [Pa]"),
    gr.Slider(0, 200, step=1, label="Downstream temperature of the GLCK [ºC]"),
    gr.Slider(0, 10, step=0.5, label="Gas lift flow rate [m³/s]"),
]
outputs = [gr.Label(num_top_classes=9)]

examples = [
    ["2017-02-01 02:02:07.000000", 0.0, 10092110.0, 119.0944, 1609800.0, 84.59782, 1564147.0, 0.0, 0.0],
    ["2014-01-20 22:05:01.000000", 0.0, 16346970.0, 117.5749, 7492611.0, 173.09610, 4438899.0, 0.0, 0.0],
]

title = "Flow Instability Anomaly Classification"
description = "Enter the details to correctly identify the Flow Instability anomaly"
article = "This app is a part of the Beginner's Guide to CI/CD for Machine Learning. It teaches how to automate training, evaluation, and deployment of models to Hugging Face using GitHub Actions."

gr.Interface(
    fn=predict_class,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
