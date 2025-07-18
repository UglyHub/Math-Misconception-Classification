# Math Misconception Classification

This project aims to classify student explanations for math problems based on predefined categories and misconceptions using a transformer-based model. The goal is to identify the specific type of understanding or misunderstanding a student has, providing valuable insights for educational assessment and intervention.

The solution utilizes the Hugging Face transformers library to fine-tune a pre-trained language model on a dataset of student responses, achieving multi-label classification where each response can be associated with one or more Category:Misconception labels.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Setup](#setup)
3.  [Data](#data)
4.  [Model](#model)
5.  [Training](#training)
6.  [Prediction](#prediction)
7.  [Submission](#submission)
8.  [Evaluation](#evaluation)
9.  [Skills Highlighted](#skills-highlighted)

## Introduction

Accurately identifying student misconceptions is crucial for personalized learning and effective teaching. This notebook demonstrates a natural language processing approach to tackle this challenge. By leveraging the power of transformer models, we can analyze student explanations and classify them into specific categories and misconceptions. The output is formatted for submission to a competition requiring Mean Average Precision at 3 (MAP@3).

## Setup

To run this notebook, you will need a Google Colab environment with GPU access enabled.

1.  *Open the notebook in Google Colab.*
2.  *Ensure GPU acceleration is enabled:* Go to Runtime > Change runtime type and select GPU under Hardware accelerator.
3.  *Install necessary libraries:* The notebook includes a cell (!pip install ...) to install all required libraries, including transformers, torch, pandas, numpy, sklearn, datasets, evaluate, accelerate, and sentencepiece.
4.  *Upload data files:* You need to upload the train.csv, test.csv, and sample_submission.csv files to your Colab environment. You can do this using the file explorer on the left sidebar.
5.  *Kaggle API Setup (for submission):* If you intend to submit your predictions to a Kaggle competition, you will need to set up the Kaggle API.
    * Generate a Kaggle API token (kaggle.json) from your Kaggle account settings.
    * Upload the kaggle.json file to your Colab environment.
    * Run the provided cells to set up the Kaggle API client.

## Data

The dataset consists of student responses to math problems. The training data (train.csv) includes:

* row_id: Unique identifier for each row.
* QuestionId: Identifier for the math question.
* QuestionText: The text of the math question.
* MC_Answer: The multiple-choice answer provided by the student.
* StudentExplanation: The student's written explanation for their answer.
* Category: The primary category of the student's response (e.g., True_Correct, False_Misconception).
* Misconception: The specific misconception identified in the student's explanation (present for some categories).

The test data (test.csv) has the same columns except for Category and Misconception.

Preprocessing steps in the notebook include:

* Combining QuestionText, MC_Answer, and StudentExplanation into a single text input for the model, using [SEP] tokens as separators.
* Creating a combined target label by concatenating Category and Misconception (handling missing Misconception values with 'NA').
* Encoding these combined labels into numerical IDs.

## Model

The project uses a pre-trained transformer model from the Hugging Face library. Specifically, the notebook uses `microsoft/deberta-v3-small`, although this can be easily changed to other compatible models like `roberta-base`.

The model is configured for multi-label classification, as a student response could potentially be associated with multiple issues (though the primary focus is on the provided Category:Misconception pairs).

## Training

The model is fine-tuned on the preprocessed training data using the Hugging Face `Trainer` API.

Key aspects of the training process:

* *Custom Dataset:* A `MathMisconceptionDataset` class is used to prepare the data for the trainer, handling tokenization and label formatting.
* *Training Arguments:* `TrainingArguments` are defined to configure the training process, including parameters for epochs, batch size, learning rate, weight decay, logging, and evaluation strategy. Mixed precision training (fp16) is enabled for efficiency on supported GPUs.
* *Evaluation:* The notebook includes a custom `compute_metrics` function to calculate the Mean Average Precision at 3 (MAP@3), the primary evaluation metric for this task. Evaluation is performed periodically during training (if configured).

## Prediction

After training, the fine-tuned model is used to make predictions on the test dataset.

* The test data is processed similarly to the training data, without the need for labels.
* The `Trainer.predict` method is used to get the raw logits from the model.
* Sigmoid activation is applied to the logits to obtain probabilities for each possible `Category:Misconception` label.
* For each test sample, the top 3 labels with the highest predicted probabilities are selected.

## Submission

The final step is to generate a submission file in the required format (`submission.csv`).

* A pandas DataFrame is created with `row_id` from the test data and the top 3 predicted `Category:Misconception` labels (space-separated) for each row.
* This DataFrame is saved to a CSV file named `submission.csv`.

## Evaluation

The performance of the model is evaluated using the Mean Average Precision at 3 (MAP@3) metric. The custom `calculate_map_at_3` function implements this metric. During training (if evaluation is enabled), this metric is computed on the validation set.

## Skills Highlighted

This project showcases a comprehensive set of skills essential for modern machine learning and data science, particularly in Natural Language Processing:

* **Natural Language Processing (NLP):** Core understanding and application of NLP techniques for text preprocessing, tokenization, and text representation.
* **Deep Learning:** Practical experience with deep learning model architectures, specifically transformer networks (e.g., DeBERTa-v3-Small).
* **Hugging Face Transformers Library:** Proficient use of the Hugging Face ecosystem for model loading, tokenizer management, fine-tuning with `Trainer` API, and handling `AutoModelForSequenceClassification`.
* **Multi-label Classification:** Expertise in setting up and solving multi-label classification problems, including appropriate loss functions (`BCEWithLogitsLoss`) and performance metrics (MAP@3).
* **Python Programming:** Strong foundational and applied Python skills for data manipulation (`pandas`), numerical operations (`numpy`), and general scripting.
* **Data Preprocessing & Feature Engineering:** Ability to clean, combine, and transform raw textual and categorical data into a format suitable for machine learning models, including handling missing values.
* **Machine Learning Workflow Management:** Experience in structuring an end-to-end machine learning project, from data ingestion and preprocessing to model training, prediction, and submission generation.
* **Google Colab & GPU Utilization:** Practical skills in leveraging cloud-based GPU environments for accelerated model training, including memory management and runtime configuration.
* **Scikit-learn:** Application of `MultiLabelBinarizer` for target label encoding and other utility functions.
* **PyTorch:** Underlying understanding of PyTorch for custom dataset creation and tensor operations.
