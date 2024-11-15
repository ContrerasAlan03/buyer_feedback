#Configuración del entorno
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

#cargo el modelo y el tokenizador de Hugging Face

sentimientos = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(sentimientos)
modelo = AutoModelForSequenceClassification.from_pretrained(sentimientos)

# se crea el pipeline para el análisis de Percepción de los consumidores:
analisis_sentimiento = pipeline("sentiment-analysis", model = modelo, tokenizer=tokenizer)

#comentarios de ejemplo
comentarios = [ 
    "I love this product! It has exceeded my expectations.",
    "The quality is terrible. I would not recommend it.",
    "It’s okay, but not what I expected.",
    "Absolutely amazing! Worth every penny.",
    "Not happy with this purchase at all."]

#Análisis de los Feedback de los consumidores:
resultados = analisis_sentimiento(comentarios)

#Visualización de los resultados:
for comentario, resultado in zip(comentarios, resultados):
    print(f"comentario :{comentario}\n sentimiento :{resultado['label']} con confianza de {resultado['score']:.2f}\n")
    
