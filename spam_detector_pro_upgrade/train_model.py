import pandas as pd
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# Preprocess
ps = PorterStemmer()
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords.words("english")]
    return " ".join(words)

df["cleaned"] = df["message"].apply(clean_text)

# Train model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["cleaned"])
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
