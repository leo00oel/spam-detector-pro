from flask import Flask, request, render_template, session
import pickle

app = Flask(__name__)
app.secret_key = 'spam-secret-key'

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'history' not in session:
        session['history'] = []

    result = None
    confidence = None
    message = ""

    if request.method == 'POST':
        message = request.form['message']
        vec = vectorizer.transform([message])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        confidence = round(max(prob) * 100, 2)
        result = "SPAM" if prediction else "NOT SPAM"

        # Add to session history
        session['history'].append({
            'text': message,
            'result': result,
            'confidence': confidence
        })
        session.modified = True

    return render_template("index.html", result=result, confidence=confidence, message=message, history=session['history'])

if __name__ == "__main__":
    app.run(debug=True)
