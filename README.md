# AI-Powered Fake News Detection System (NLP + ML + Flask)

An end-to-end Fake News Detection system built using Natural Language Processing (NLP), Machine Learning models, and Flask for deployment.  
This project detects whether a political news article is **Real** or **Fake** based on text patterns learned during model training.

---

## Features

### Fake & Real News Classification  
Classifies news text into:
- **REAL NEWS**
- **FAKE NEWS**

### NLP Pipeline  
Includes:
- Text cleaning  
- Stopword removal  
- Lemmatization  
- TF-IDF vectorization  
- Dimensionality reduction with SVD  
- MLP Neural Network classification  

### Trained ML Models  
Stored in the `model/` folder:
- `tfidf_vectorizer.joblib`
- `svd_300.joblib`
- `TFIDF_MLP.joblib` (best model)

### Flask Web Application  
Simple UI where users can paste news text and get predictions instantly.

---

## Project Structure

AI-Powered-Fake-News-Detection-System/
│
├── app.py
├── model/
│ ├── tfidf_vectorizer.joblib
│ ├── svd_300.joblib
│ └── TFIDF_MLP.joblib
├── templates/
│ └── index.html
├── app/
│ └── style.css
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ └── 02_model_training.ipynb
├── requirements.txt
└── README.md


---

## Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **TF-IDF + SVD + MLP** | **0.9890** | **0.9882** |
| TF-IDF + LinearSVC | 0.9833 | 0.9839 |
| TF-IDF + Logistic Regression | 0.9782 | 0.9791 |
| Word2Vec + MLP | 0.9723 | 0.9731 |

**Best model:**  
⭐ **TF-IDF + SVD + MLP Neural Network**

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-Learn  
- Gensim (Word2Vec)  
- Flask  
- HTML + CSS  
- Gunicorn  

---

## 🖥 How to Run Locally

### 1️⃣ Clone the repository
git clone https://github.com/boosireddyprasadreddy7/AI-Powered-Fake-News-Detection-System.git

cd AI-Powered-Fake-News-Detection-System

### 2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Run the Flask app
python app.py


---

## 🌐 Deployment (Render)

1. Push your project to GitHub  
2. Go to https://render.com  
3. Create a **New Web Service**  
4. Connect your GitHub repo  
5. Use:

**Build Command**
pip install -r requirements.txt

**Start Command**
gunicorn app:app

Render will host your app online automatically.

---

## 📘 Dataset

Dataset: **Fake and Real News Dataset (Kaggle)**  
Contains:
- Fake political news  
- Real political journalism (Reuters-style)

---

## 🤝 Contributions

Pull requests are welcome.  
For major changes, please open an issue first.

---

## 📧 Contact

**Boosi Reddy Prasad Reddy**  
Email: boosireddyprasadreddy@gmail.com  
GitHub: https://github.com/boosireddyprasadreddy7 

