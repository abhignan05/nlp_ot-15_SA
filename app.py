from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        message = request.form.get("message")
        print(message)
    else:
        return render_template("predict.html")
if __name__ == "__main__":
    app.run()