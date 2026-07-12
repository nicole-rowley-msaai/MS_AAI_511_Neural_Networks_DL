# ANN Advertising Click Prediction Project Pipeline

## Project Introduction
This project focuses on a binary classification task in an advertising context: predicting whether a user will click on an online ad. The dataset includes behavioral and demographic signals such as daily time on site, age, area income, daily internet usage, ad context, location attributes, and gender.

## Project Objective
Build and optimize an Artificial Neural Network (ANN) classifier that predicts whether a user clicks an ad, then evaluate and interpret model performance using appropriate classification metrics.

## Methodology
Methodology is based on the assignment instructions and current notebook workflow:

1. Load the advertising dataset into a pandas DataFrame.
2. Perform data cleaning and preprocessing.
3. Split data into train and test sets (80:20).
4. Scale numeric features with StandardScaler.
5. Build an ANN classifier.
6. Tune architecture and training hyperparameters (layers, activations, regularization, learning rate, batch size).
7. Evaluate with accuracy, precision, recall, F1, and ROC-AUC.
8. Interpret findings and identify key predictive factors.

## End-to-End Project Pipeline

### Phase 1: Problem Framing and Success Criteria
- Define business question: predict ad click likelihood.
- Define primary success metric: ROC-AUC (with F1 and recall support).
- Define minimum acceptance thresholds for deployment readiness.

### Phase 2: Data Ingestion and Validation
- Load zipped CSV data.
- Validate schema, column types, target distribution, and missing values.
- Detect leakage-prone fields (for example, time or text IDs if not engineered properly).

### Phase 3: Data Preparation
- Drop or transform non-model-ready columns (Timestamp, Ad Topic Line unless engineered).
- Encode categorical features (one-hot encoding for City and Country).
- Separate features (X) and target (y).
- Split train/test with fixed random seed for reproducibility.
- Fit StandardScaler on train only; transform train and test.

### Phase 4: Baseline Modeling
- Build a simple ANN baseline (input -> hidden -> hidden -> sigmoid output).
- Train with binary cross-entropy compatible output.
- Record baseline metrics and confusion matrix.

### Phase 5: Controlled Experiments
- Run experiment grid across:
  - Hidden layer sizes and depth.
  - Activation functions (ReLU, LeakyReLU, Tanh).
  - Regularization (dropout, L2/weight decay).
  - Learning rates and optimizers (Adam, RMSprop).
  - Batch sizes and epochs.
- Track each run in an experiment table (config, validation metrics, notes).

### Phase 6: Evaluation and Error Analysis
- Evaluate best candidate on hold-out test set.
- Report accuracy, precision, recall, F1, ROC-AUC.
- Plot ROC curve and inspect confusion matrix.
- Perform error analysis by feature slices (age groups, income bands, usage levels).

### Phase 7: Interpretation and Insights
- Identify most influential predictors using:
  - Permutation importance on processed features, or
  - SHAP (if added) for local/global explanation.
- Translate technical outputs into advertising strategy recommendations.

### Phase 8: Reproducibility and Packaging
- Save preprocessing objects (encoder/scaler) and final model.
- Export inference-ready pipeline steps and model metadata.
- Document environment, seed values, and run configuration.

### Phase 9: Delivery Artifacts
- Final notebook with clean narrative and results.
- Summary report with metrics, visualizations, and recommendations.
- Optional lightweight inference script for new user records.

## Immediate Execution Checklist
- Confirm dataset path is portable (avoid user-specific absolute path).
- Wrap preprocessing and model training into reusable functions.
- Add a single experiment log table for all tuning runs.
- Add ROC curve, confusion matrix, and feature-importance analysis.
- Produce final model card: data, metrics, limitations, and next steps.
