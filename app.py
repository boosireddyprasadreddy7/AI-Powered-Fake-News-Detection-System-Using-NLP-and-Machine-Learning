from flask import Flask, render_template, request
import joblib
import os

# Load model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
svd = joblib.load(os.path.join(MODEL_DIR, "svd_300.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "TFIDF_MLP.joblib"))

app = Flask(__name__)

def predict_news(text):
    x_tfidf = tfidf.transform([text])
    x_svd = svd.transform(x_tfidf)
    pred = model.predict(x_svd)[0]
    return "REAL NEWS" if pred == 1 else "FAKE NEWS"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("news_text", "").strip()
        if input_text:
            prediction = predict_news(input_text)
        else:
            prediction = "Please enter some text."

    return render_template("index.html", prediction=prediction, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
