from data import faq_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Convert FAQ data into a list of questions and answers
questions = list(faq_data.keys())
answers = list(faq_data.values())

# Fit the vectorizer on the FAQ questions
vectorizer = TfidfVectorizer().fit(questions)

def get_answer(user_question):
    # Transform user question and FAQ questions into TF-IDF vectors
    user_question_tfidf = vectorizer.transform([user_question])
    faq_questions_tfidf = vectorizer.transform(questions)

    # Calculate cosine similarity between user question and each FAQ question
    similarities = cosine_similarity(user_question_tfidf, faq_questions_tfidf)

    # Find the index of the most similar FAQ question
    best_match_index = np.argmax(similarities, axis=1)[0]

    # Return the answer corresponding to the best matching FAQ question
    return answers[best_match_index]


# Test the chatbot
if __name__ == "__main__":
    print("Chatbot: Hello! Ask me anything about Python.")
    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        else:
            response = get_answer(user_question)
            print("Chatbot:", response)
