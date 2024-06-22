from flask import Flask, render_template, request
import joblib
import re
from nltk.tokenize import word_tokenize
# Loading the model and vectorizer using joblib
app = Flask(__name__)
model = joblib.load("hausa_sentiment_model.pkl")
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# my created Custom Hausa stopwords
hausa_stopwords = set([
    'da', 'shi', 'su', 'wani', 'yana', 'daga', 'sai', 'cikin', 'kai', 'gare',
    'amma', 'ta', 'in', 'lokacin', 'kuma', 'idan', 'daga', 'kusa', 'zata',
    'tafi', 'zata', 'don', 'don', 'kuma', 'haka', 'yau', 'yazo', 'kwata',
    'kuma', 'ta', 'ta', 'zata', 'id', 'dawo', 'tare', 'ya', 'shi', 'kira',
    'biyu', 'har', 'bata', 'ba', 'bamu', 'kai', 'mata', 'mu', 'ku', 'shin',
    'ki', 'su', 'lokacin', 'a', 'an', 'da', 'sai', 'da', 'an', 'sai', 'daga',
    'koda', 'haka', 'yau', 'in', 'lokacin', 'wacce', 'kuma', 'yap', 'haka',
    'tafi', 'tafi', 'ya', 'daga', 'ya', 'ta', 'ta', 'id', 'tare', 'da', 'ta',
    'a', 'kafin', 'kuma', 'kamar', 'kan', 'kada', 'kawai', 'zata', 'an',
    'wannan', 'yana', 'kan', 'suka', 'sun', 'sunan', 'sunansa', 'tafarki',
    'tafarkin', 'tunda', 'ta', 'tafi', 'tafi', 'tare', 'da', 'ta', 'ta',
])

# Preprocessing the text , removing non-alphabetice characters
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabet characters
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_tokens = [token for token in tokens if
                       token.isalpha() and token not in hausa_stopwords]  # Remove stopwords and stem
    return " ".join(filtered_tokens)
@app.route('/')
def index():
    # landing page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # getting the text from form
        mytext = request.form.get('text')

        # appplying preprocessing on the text
        cleaned_text = preprocess_text(mytext)

        # converrting the preprocess text to tfidf array
        tfidf_features = tfidf_vectorizer.transform([cleaned_text]).toarray()

        # making predicitons
        prediction = model.predict(tfidf_features)[0]

        # Mapping the prediction with the below labels, 1=positive 0==
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        sentiment = sentiment_labels[prediction]

        # Calculate probabilities for all classes
        probabilities = model.predict_proba(tfidf_features)
        probability = probabilities[0]
        if probability[1]<=0.65:
            if probability[1]>probability[0] and probability[1]>probability[2] :
                if probability[0]>probability[2]:
                    sentiment="negaitive"
                    probability[1]=probability[1]-probability[0]
                    probability[0]=probability[0]+probability[1]-probability[2]
                elif probability[2]>probability[0]:
                    sentiment="positive"
                    probability[1]=probability[1]-probability[2]
                    probability[2] = probability[2] + probability[1] - probability[0]
        else:
            sentiment=sentiment
        probability_formatted = [f'{p:.2f}' for p in probability]

        return render_template('index.html', text=mytext, prediction=sentiment, probability=probability_formatted)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)