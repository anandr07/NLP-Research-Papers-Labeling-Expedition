#%%
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import matplotlib.pyplot as plt

#%%
# Function to extract non-alphanumeric characters from a string
def extract_non_alphanumeric(text):
    non_alphanumeric = re.findall(r'[^a-zA-Z0-9\s]', text)
    return non_alphanumeric

#%%
# Define the clean_text function
def clean_text(text):
    ''' This function removes punctuations, HTML tags, URLs, and Non-Alphanumeric words.
    '''
    unwanted_chars_patterns = [
        r'[!?,;:—".]',  # Remove punctuation
        r'<[^>]+>',  # Remove HTML tags
        r'http[s]?://\S+',  # Remove URLs
        r'\W',  # Non-Alphanumeric
    ]
    
    for pattern in unwanted_chars_patterns:
        text = re.sub(pattern, '', text)
    
    return text

#%%
# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
nltk.download('punkt')

#%%
def preprocess_text(text):
    ''' This function performs tokenization of text and also uses Snowball Stemmer for stemming of words.
    '''
    if isinstance(text, str):  # Check if text is a string
        # Tokenizing the text and removing stopwords
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and word.isalpha() and len(word) >= 3]
        # Applying Snowball stemming
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)
    else:
        return ''  # Return an empty string for non-string inputs

#%%
# Read the data
train_data = pd.read_csv(r"C:\Anand\Projects_GWU\NLP-Research-Paper-Project\NLP-Research-Labeling-Expedition-main\NLP-Research-Labeling-Expedition-main\data\train.csv")

#%%
# Clean the data
train_data_cleaned = train_data.copy()  # Create a copy of the original DataFrame
train_data_cleaned['abstract'] = train_data_cleaned['abstract'].apply(preprocess_text)

#%%
# Tokenize the cleaned abstracts into words without stemming
abstract_words = [word for abstract in train_data_cleaned['abstract'] for word in abstract.split()]

#%%
# Count the frequency of each word before stemming
word_freq = Counter(abstract_words)

#%%
# Sort the words based on their frequency
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

#%%
# Display the top N most used words before stemming
top_n = 30  # Change this value to display more or fewer top words
print(f"Top {top_n} most used words in the 'abstract' column before stemming:")
for word, freq in sorted_word_freq[:top_n]:
    print(f"{word}: {freq}")

#%%
# Get the top 30 most used words and their frequencies before stemming
top_words = [word[0] for word in sorted_word_freq[:30]]
word_frequencies = [word[1] for word in sorted_word_freq[:30]]

#%%
# Calculate the total number of words before stemming
total_words = sum(word_frequencies)

#%%
# Calculate the percentages rounded to one decimal precision before stemming
word_percentages = [(freq / total_words) * 100 for freq in word_frequencies]
word_percentages_rounded = [round(percentage, 1) for percentage in word_percentages]

#%%
# Plotting the histogram
plt.figure(figsize=(12, 6))
plt.bar(range(len(top_words)), word_percentages, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Percentage')
plt.title('Top 30 Most Used Words in Abstracts (Before Stemming)')
plt.xticks(range(len(top_words)), top_words, rotation=90)

# Show percentage on top of bars
for i, percentage in enumerate(word_percentages_rounded):
    plt.text(i, percentage + 0.5, f'{percentage}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

#%%
