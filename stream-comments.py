# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the preprocessed data
data_clean = pd.read_csv('data/databersih1.csv', encoding='latin1')
sentiment_data = pd.read_csv('data/data.csv', sep=';', encoding='utf-8')[['Sentimen']]
data_combined = pd.concat([data_clean, sentiment_data], axis=1)
data_combined['cleaned_text'].fillna('', inplace=True)

# Load the model and vectorizer outside the main function
naive_bayes = MultinomialNB()
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(data_combined['cleaned_text'])
y_train = data_combined['Sentimen']
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_vec, y_train)

# Train the model
naive_bayes.fit(x_train_resampled, y_train_resampled)

def main():
    st.title("Sentiment Analysis App")

    # User input for comment
    comment_input = st.text_area("Enter your comment:")

    if st.button("Analyze Sentiment"):
        # Perform sentiment analysis
        sentiment_label = analisis_komentar(comment_input)
        st.write(f"Sentiment: {sentiment_label}")

        # Display suggestions if sentiment is negative
        if sentiment_label == "Pengunjung Tidak Puas":
            st.write("Suggestions for Improvement:")
            suggestions = saran_perbaikan(comment_input)
            for idx, suggestion in enumerate(suggestions, start=1):
                st.write(f"{idx}. {suggestion}")

        # Display WordCloud for the entered comment
        visualize_wordcloud(comment_input)

# Function to perform sentiment analysis
def analisis_komentar(comment):
    comment_input_vec = vectorizer.transform([comment])
    predicted_sentimen = naive_bayes.predict(comment_input_vec)

    if predicted_sentimen[0] == 1:
        sentiment_label = "Pengunjung Puas"
    else:
        sentiment_label = "Pengunjung Tidak Puas"

    return sentiment_label

# Function to provide improvement suggestions
def saran_perbaikan(comment):
    sentiment = analisis_komentar(comment)
    list_saran = []

    if sentiment == "Pengunjung Tidak Puas":
        if "bau" in comment.lower():
            list_saran.append("Perlu peningkatan kebersihan atau ventilasi di area mall.")
        if "toilet" in comment.lower():
            list_saran.append("Perlu peningkatan fasilitas dan kebersihan pada toilet mall")
        if "antri" in comment.lower():
            list_saran.append("Perlu koordinasi serta peningkatan fasilitas mall agar tidak terjadi antrian")
        if "kotor" in comment.lower():
            list_saran.append("Perlu penambahan petugas kebersihan dan fasilitas pendukung untuk kebersihan Mall PVJ")
        if "musholla" in comment.lower():
            list_saran.append("Perlu peningkatan fasilitas untuk Mushola mall PVJ")
        return list_saran
    else:
        return ["Tidak ada hal yang harus di perbaiki, Terima kasih atas komentar Anda!"]  # Jika puas, kembalikan pesan ini

# Function to visualize WordCloud
def visualize_wordcloud(comment):
    plt.figure(figsize=(10, 5))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comment)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud for Entered Comment")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
