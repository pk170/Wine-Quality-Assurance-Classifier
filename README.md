# Wine Quality Analyser

## 📌 Project Overview
This project implements a **Random Forest Classifier** to automate quality control testing. Using physicochemical data (such as acidity, pH, and residual sugar), the model predicts whether a product meets "Premium" quality standards or falls into the "Standard" category. 

## Tech Stack
* **Python 3.x**
* **Scikit-Learn** (Random Forest, Classification Metrics)
* **Matplotlib** (Feature Importance Visualisation)
* **Pandas** (Data Manipulation)

## Methodology
1. **Binary Classification:** Converted the raw 0-10 quality score into a binary "Pass/Fail" metric (Quality ≥ 7 is Premium).
2. **Feature Scaling:** Applied `StandardScaler` to normalise chemical measurements.
3. **Modeling:** Trained a Random Forest Ensemble to handle non linear relationships between chemical properties.
4. **Analysis:** Extracted **Feature Importance** to identify which chemical factors (e.g., Alcohol level, Sulphates) most strongly influence the final quality rating.

## Key Results
* **Accuracy:** Achieved ~88-90% classification accuracy on the test set.
* **Key Insight:** Alcohol content and Sulphates were identified as the strongest indicators of premium quality.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the classifier: `python quality_analyser_engine.py`
