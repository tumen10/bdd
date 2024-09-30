from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка датасета
dataset = load_dataset("deepvk/ru-HNP")

# Генерация векторных представлений
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(dataset['train']['text'])

# Функция для оценки сходства
def calculate_similarity(text1, text2):
    vector1 = vectorizer.transform([text1])
    vector2 = vectorizer.transform([text2])
    return cosine_similarity(vector1, vector2)

# Функция для масштабирования сходства
def scale_similarity(similarity):
    return (similarity * 10) - 5

# Проведение экспериментов
similarities = []
for i in range(len(dataset['train']['text'])):
    for j in range(i+1, len(dataset['train']['text'])):
        similarity = calculate_similarity(dataset['train']['text'][i], dataset['train']['text'][j])
        scaled_similarity = scale_similarity(similarity)
        similarities.append((dataset['train']['text'][i], dataset['train']['text'][j], scaled_similarity))

# Вывод результатов
for text1, text2, similarity in similarities:
  print(f"Text 1: {text1}, Text 2: {text2}, Similarity: {similarity}")

