# Practical Machine Learning: Barbell Lift Exercise Manner Prediction

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-View%20Analysis-blue.svg)](https://buddywhitman.github.io/pml-barbell-lift-prediction/)
[![R Markdown](https://img.shields.io/badge/Made%20with-R%20Markdown-276DC3.svg)](https://rmarkdown.rstudio.com/)

## Project Overview

This repository contains the peer-reviewed course project for the **Coursera Practical Machine Learning** course (Johns Hopkins University). 

Using accelerometer data collected from the **belt, forearm, arm, and dumbbell** of 6 participants performing barbell lifts, the goal is to predict the manner in which the exercise was performed (`classe` variable):
- **Class A**: Exactly according to the specification (Correct execution)
- **Class B**: Throwing the elbows to the front
- **Class C**: Lifting the dumbbell only halfway
- **Class D**: Lowering the dumbbell only halfway
- **Class E**: Throwing the hips to the front

The dataset comes from the Human Activity Recognition (HAR) study by Velloso, E. et al. (2013).

---

## Key Results & Findings

- **Primary Model**: Random Forest (`randomForest`) trained on 52 preprocessed accelerometer features across 13,737 training samples (70% partition).
- **Validation Performance**: Evaluated on an independent holdout set of 5,885 observations (30% partition).
  - **Validation Accuracy**: **99.52%**
  - **Expected Out-of-Sample Error**: **0.48%** ($1 - \text{Accuracy}$)
  - **Out-of-Bag (OOB) Error Rate**: **0.54%**
- **Baseline Comparison**: Outperformed a single Decision Tree baseline (`rpart`), which achieved 73.59% accuracy.

---

## 20 Course Project Test Case Predictions

Applying the final Random Forest model to the 20 test cases (`pml-testing.csv`) yielded the following predictions:

| Problem ID | Predicted `classe` |
| :---: | :---: |
| 1 | **B** |
| 2 | **A** |
| 3 | **B** |
| 4 | **A** |
| 5 | **A** |
| 6 | **E** |
| 7 | **D** |
| 8 | **B** |
| 9 | **A** |
| 10 | **A** |
| 11 | **B** |
| 12 | **C** |
| 13 | **B** |
| 14 | **A** |
| 15 | **E** |
| 16 | **E** |
| 17 | **A** |
| 18 | **B** |
| 19 | **B** |
| 20 | **B** |

---

## Interactive HTML Report

You can view the full compiled report online via GitHub Pages:
👉 **[View Compiled HTML Report](https://buddywhitman.github.io/pml-barbell-lift-prediction/)**

Alternatively, clone this repository and open [`index.html`](file:///D:/OneDrive/Documents/repos/courseraproject4/index.html) in any web browser.

---

## Reproducing the Analysis

To re-run the analysis and re-compile the report:

1. Clone the repository:
   ```bash
   git clone https://github.com/buddywhitman/pml-barbell-lift-prediction.git
   cd pml-barbell-lift-prediction
   ```

2. Open R / RStudio and ensure dependencies are installed:
   ```R
   install.packages(c("caret", "randomForest", "rpart", "rpart.plot", "corrplot", "rmarkdown", "knitr"))
   ```

3. Render the R Markdown document:
   ```R
   rmarkdown::render("index.Rmd", output_file = "index.html")
   ```

---

## Citation

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **Qualitative Activity Recognition of Weight Lifting Exercises**. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.
