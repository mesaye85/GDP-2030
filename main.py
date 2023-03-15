from flask import Flask, render_template, request, send_file
import io
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    growth_rate = float(request.form["growth_rate"])

    # Prepare the input features for prediction
    future_year = X['year'].max() + 1
    future_data = pd.DataFrame({'year': [future_year] * len(X['country_code'].unique()), 'country_code': X['country_code'].unique()})

    # Predict future GDP
    future_gdp = model.predict(future_data) * (1 + growth_rate)

    # Plot the result
    plt.plot(future_gdp)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.clf()

    # Encode the image as base64
    img_base64 = base64.b64encode(img.getvalue()).decode()

    return render_template("result.html", img_base64=img_base64)

if __name__ == "__main__":
    app.run()
