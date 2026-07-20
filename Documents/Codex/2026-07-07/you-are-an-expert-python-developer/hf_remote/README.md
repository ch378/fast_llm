---
title: Banking CSV Evaluation
emoji: "📊"
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
---

# Banking CSV Evaluation

This Space evaluates extracted banking-fee CSV files for:

- expected columns;
- provider consistency;
- page values;
- numeric rates and `Variable` rates;
- duplicate rows;
- rows requiring review.

It also loads the included `model.pkl` XGBoost quality model for Mastercard rows.

The Space does not contain proprietary PDFs or API keys. Upload a CSV at runtime.
