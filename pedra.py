from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

curriculo = """
Estudante de engenharia com experiência em Python,
análise de dados, Excel, machine learning.
"""

vaga = """
Procuramos alguém com conhecimento em Python,
análise de dados, SQL e dashboards.
"""

vectorizer = CountVectorizer()
vetores = vectorizer.fit_transform([curriculo, vaga])

similaridade = cosine_similarity(vetores)[0][1]

print(f"Compatibilidade: {round(similaridade * 100, 2)}%")