from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    growth_rate = float(request.form["growth_rate"])
    future_gdp = model.predict(X) * (1 + growth_rate)

    plt.plot(future_gdp)
    plt.show()

    return "Prediction complete"

if __name__ == "__main__":
    app.run()
