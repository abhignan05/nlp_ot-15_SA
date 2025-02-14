from flask import Flask, render_template, request
from test import TextToNum
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        print(f"User Input: {msg}")

        # Preprocess the input message
        cl = TextToNum(msg)
        cl.cleaner()
        cl.token()
        cl.removeStop()
        stemmed_words = cl.stemme()
        processed_text = " ".join(stemmed_words)

        # Load vectorizer
        with open("vectorizer.pickle", "rb") as vc_file:
            vectorizer = pickle.load(vc_file)
        
        dt = vectorizer.transform([processed_text]).toarray()

        # Load model
        with open("model.pickle", "rb") as mb_file:
            model = pickle.load(mb_file)

        # Make prediction
        pred = model.predict(dt)[0]  # Extract the single prediction

        # Convert numeric prediction to text labels
        if pred == -1:
            pred = "Negative"
        elif pred == 0:
            pred = "Neutral"
        else:
            pred = "Positive"

        print(f"Prediction: {pred}")

        return render_template("result.html", prediction=pred)
    
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5080)
