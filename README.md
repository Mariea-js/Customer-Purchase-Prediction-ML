# Customer Purchase Prediction — Data Preparation & PyTorch Classification

An educational machine-learning portfolio project demonstrating customer-session data cleaning, numeric feature scaling, categorical encoding, and binary purchase prediction with a small PyTorch neural network.

## What the project does

1. **Cleaning (500 raw records):** standardizes missing-value markers, imputes numerical and categorical fields, and converts data types.
2. **Feature preparation (500 separate records):** applies MinMax scaling to three numerical features and one-hot encoding to two categorical features.
3. **Modelling (800 labelled records):** trains a feed-forward neural network with eight hidden units, ReLU, sigmoid output, binary cross-entropy and Adam.
4. **Added portfolio evaluation:** stratified held-out split, majority-class baseline, accuracy/precision/recall/F1, confusion matrix and training-loss visualization.
5. **Prediction (200 unlabelled records):** produces purchase predictions; the notebook can export them to `outputs/validation_predictions.csv`.

**Important:** These are three distinct educational tasks with supplied input datasets. The 500-record cleaning dataset is **not** the source of the 800-row model training dataset. The original training and validation features are supplied already encoded/scaled; their upstream transformation fitting process cannot be verified from the available files.

## Files

- `Customer_Purchase_Prediction.ipynb` — original practical-exam work plus labelled held-out evaluation and documentation
- `data/raw_customer_data.csv` — Task 1 input
- `data/model_data.csv` — Task 2 input
- `data/input_model_features.csv` — Task 3 labelled training input
- `data/validation_features.csv` — Task 3 unlabelled prediction input
- `requirements.txt` — Python dependencies

## Run

```bash
python -m pip install -r requirements.txt
jupyter notebook Customer_Purchase_Prediction.ipynb
```

Run all notebook cells from the repository root. Review the printed metrics before describing results in a CV or public post. The generated `outputs/` directory is gitignored by default.

## What is demonstrated

pandas data preparation, MinMaxScaler, one-hot encoding, PyTorch binary classification, held-out model evaluation, majority-class baseline comparison and interpretation of classification errors.

## Limitations and responsible interpretation

- Educational practical-exam data; not a production e-commerce solution.
- Class imbalance makes accuracy insufficient on its own.
- The original 200-row validation file has no purchase labels, so its prediction quality cannot be measured directly.
- `basket_value` may be a post-purchase variable, potentially creating target leakage; its timing is unverified.
- Original preprocessing provenance is unavailable for supplied model features; leakage-free upstream preparation cannot be established.
- No business ROI, causal relationship or real-world generalization is claimed.

## Attribution

The original cleaning, feature-preparation and neural-network tasks were completed as a practical examination. The portfolio extension adds reproducible evaluation, baseline comparison, visualizations, interpretation and repository documentation.
