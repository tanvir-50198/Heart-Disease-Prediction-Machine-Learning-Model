# ❤️ Heart Disease Prediction Machine Learning Web Application

Python • Scikit-Learn • Streamlit • GitHub

---

# 📌 Project Overview

The **Heart Disease Prediction Machine Learning Web Application** is an end-to-end machine learning project designed to predict whether a patient is at risk of developing heart disease based on various medical attributes.

Users can enter patient information such as age, gender, blood pressure, cholesterol level, glucose level, BMI, smoking habits, and other health indicators to receive an instant heart disease risk prediction.

🌐 **Live App:**
https://heart-disease-prediction-machine-learning-model-vrbmhep6zsbvjk.streamlit.app/

---

# 🎯 Objectives

* Build a machine learning model capable of predicting heart disease risk.
* Develop an interactive web interface for real-time predictions.
* Deploy the application online for public access.
* Gain practical experience in the complete machine learning lifecycle.

---

# 🛠️ Tech Stack

| Category         | Tools                             |
| ---------------- | --------------------------------- |
| Language         | Python                            |
| Data Processing  | Pandas, NumPy                     |
| Machine Learning | Scikit-Learn, Logistic Regression |
| Model Saving     | Pickle                            |
| Web Application  | Streamlit                         |
| Version Control  | Git, GitHub                       |
| Deployment       | Streamlit Community Cloud         |

---

# 📊 Dataset

**Dataset:** Heart Disease Dataset (`heart_disease.csv`)

**Target Variable:**

Heart Disease Risk

---

# Features Used

| Feature                  | Type        |
| ------------------------ | ----------- |
| Gender                   | Categorical |
| Age                      | Numerical   |
| Current Smoker           | Binary      |
| Cigarettes Per Day       | Numerical   |
| BP Medication            | Binary      |
| Previous Stroke          | Binary      |
| Hypertension             | Binary      |
| Diabetes                 | Binary      |
| Total Cholesterol        | Numerical   |
| Systolic Blood Pressure  | Numerical   |
| Diastolic Blood Pressure | Numerical   |
| BMI                      | Numerical   |
| Heart Rate               | Numerical   |
| Glucose                  | Numerical   |

---

# 🔄 Data Preprocessing

## Steps Performed

* Checked dataset for missing values
* Removed unnecessary data
* Verified duplicate records
* Selected relevant health features
* Prepared dataset for machine learning

---

# Categorical Encoding

| Feature         | Encoding             |
| --------------- | -------------------- |
| Gender          | Male → 1, Female → 0 |
| Current Smoker  | Yes → 1, No → 0      |
| BP Medication   | Yes → 1, No → 0      |
| Previous Stroke | Yes → 1, No → 0      |
| Hypertension    | Yes → 1, No → 0      |
| Diabetes        | Yes → 1, No → 0      |

---

# 🤖 Machine Learning Model

**Algorithm:** Logistic Regression (Scikit-Learn)

---

## Training Pipeline

Dataset Loading

↓

Data Cleaning

↓

Feature Selection

↓

Train-Test Split

↓

Model Training

↓

Model Evaluation

↓

Model Saving (`model.pkl`)

---

## Model Saving

```python
import pickle

pickle.dump(model, open("model.pkl", "wb"))
```

---

# 🌐 Web Application Features

* ✅ Interactive Streamlit Interface
* ✅ Real-Time Heart Disease Prediction
* ✅ Easy-to-use Dropdowns & Input Fields
* ✅ Instant Machine Learning Prediction
* ✅ Responsive User Interface
* ✅ Cloud Deployment using Streamlit

---

# ❤️ How to Use

1. Select Gender
2. Enter Age
3. Select Smoking Status
4. Enter Cigarettes Per Day
5. Select BP Medication Status
6. Select Stroke History
7. Select Hypertension Status
8. Select Diabetes Status
9. Enter Cholesterol Level
10. Enter Blood Pressure
11. Enter BMI
12. Enter Heart Rate
13. Enter Glucose Level
14. Click **Predict**

The application will instantly predict whether the patient is at risk of heart disease.

---

# 🚀 Deployment

**Platform:** Streamlit Community Cloud

## Steps

* Push project files to GitHub
* Connect repository with Streamlit Cloud
* Configure deployment settings
* Deploy and obtain a public URL

---

# Files Required for Deployment

* app.py
* model.pkl
* heart_disease.csv
* requirements.txt
* Heart_Disease_Prediction.ipynb

---

# requirements.txt

```text
streamlit
pandas
numpy
scikit-learn
```

---

# 💡 Key Learning Outcomes

* Building an end-to-end Machine Learning pipeline
* Developing interactive web applications with Streamlit
* Deploying ML applications to Streamlit Cloud
* Managing project dependencies
* Creating user-friendly healthcare prediction systems

---

# 📁 Project Structure

```
Heart-Disease-Prediction-Machine-Learning-Model/
│
├── app.py
├── Heart_Disease_Prediction.ipynb
├── heart_disease.csv
├── model.pkl
├── requirements.txt
├── README.md
```


---

# ▶️ Run Locally

```bash
# Clone Repository

git clone https://github.com/tanvir-50198/Heart-Disease-Prediction-Machine-Learning-Model.git

# Navigate

cd Heart-Disease-Prediction-Machine-Learning-Model

# Install Dependencies

pip install -r requirements.txt

# Run

streamlit run app.py
```

---

# 👨‍💻 Author

**Tanvir Rahman**

Computer Science & Engineering Student

🌐 **Live App**
https://heart-disease-prediction-machine-learning-model-vrbmhep6zsbvjk.streamlit.app/

💻 **GitHub**
https://github.com/tanvir-50198

💼 **LinkedIn**
https://www.linkedin.com/in/tanvir-rahman-50198/

---

# 📄 License

This project is open source and available for educational and learning purposes.

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Thank you for visiting this repository!
