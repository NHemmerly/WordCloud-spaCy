import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

filepath = "../textInput/MTGMonsterVid.txt"

with open(filepath, 'r') as file:
    content = file.read()

nlp = spacy.load("en_core_web_sm")

doc = nlp(content)

filtered_tokens = [token.text for token in doc if not token.is_stop]

filtered_text = ' '.join(filtered_tokens)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()