#%%
# Importing Libraries 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from wordcloud import WordCloud


#%%
train_data = pd.read_csv("data\\train.csv")

#%%
# Analyzing the DataFrame : Viewing the Data

train_data.head()

#%%
# Info about the data

train_data.info()

#%%
# Cleaning Data : Checking for Empty Cells

train_data.isnull().values.sum()

#%%
# Checking for Duplicate Cells

train_data.duplicated().sum()

#%%
# Column Data Type Assessment

train_data.dtypes.value_counts()

# %%
# Function to extract non-alphanumeric characters from a string
def extract_non_alphanumeric(text):
    non_alphanumeric = re.findall(r'[^a-zA-Z0-9\s]', text)
    return non_alphanumeric

# List to store non-alphanumeric characters
non_alphanumeric_list = []

# Iterate over each title in the "title" column of train_data
for title in train_data['title']:
    non_alphanumeric_list.extend(extract_non_alphanumeric(title))

# Remove duplicates
non_alphanumeric_list = list(set(non_alphanumeric_list))

print("List of non-alphanumeric characters:", non_alphanumeric_list)
print("Number of non-alphanumeric characters:", len(non_alphanumeric_list))

# %%
# Function to extract non-alphanumeric characters from a string
def extract_non_alphanumeric(text):
    non_alphanumeric = re.findall(r'[^a-zA-Z0-9\s]', text)
    return non_alphanumeric

# List to store non-alphanumeric characters
non_alphanumeric_list = []

# Iterate over each abstract in the "abstract" column of train_data
for abstract in train_data['abstract']:
    if isinstance(abstract, str):  # Check if the abstract is a string
        non_alphanumeric_list.extend(extract_non_alphanumeric(abstract))

# Remove duplicates
non_alphanumeric_list = list(set(non_alphanumeric_list))

print("List of non-alphanumeric characters in the abstract column:", non_alphanumeric_list)
print("Count of non-alphanumeric characters in the abstract column:", len(non_alphanumeric_list))

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

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
nltk.download('punkt')

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

# Define the sentence_to_words function
def sentence_to_words(data_frame, column_name):
    ''' This function converts a sentence into words keeping words that are alphanumeric only.
        Also makes all the words lowercase.
    '''
    list_of_words_in_sentence = []

    for sent in data_frame[column_name].values:
        sent = clean_text(sent)
        # Split the sentence into words and keep only alphanumeric words
        words = [word.lower() for word in sent.split() if word.isalnum()]
        list_of_words_in_sentence.append(words)

    return list_of_words_in_sentence

# Apply text processing to the "abstract" column of train_data
train_data_cleaned = train_data.copy()  # Create a copy of the original DataFrame
train_data_cleaned['abstract'] = train_data_cleaned['abstract'].apply(preprocess_text)

# Save the cleaned dataset as train_data_cleaned
# train_data_cleaned.to_csv('train_data_cleaned.csv', index=False)


# %%
print(train_data_cleaned["abstract"].head())

# %%
# Tokenize the cleaned abstracts into words
abstract_words = [word for abstract in train_data_cleaned['abstract'] for word in abstract.split()]

# Count the frequency of each word
word_freq = Counter(abstract_words)

# Sort the words based on their frequency
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Display the top N most used words
top_n = 30  # Change this value to display more or fewer top words
print(f"Top {top_n} most used words in the 'abstract' column:")
for word, freq in sorted_word_freq[:top_n]:
    print(f"{word}: {freq}")

# %%
# Get the top 30 most used words and their frequencies
top_words = [word[0] for word in sorted_word_freq[:30]]
word_frequencies = [word[1] for word in sorted_word_freq[:30]]

# Calculate the total number of words
total_words = sum(word_frequencies)

# Calculate the percentages rounded to one decimal precision
word_percentages = [(freq / total_words) * 100 for freq in word_frequencies]
word_percentages_rounded = [round(percentage, 1) for percentage in word_percentages]

# Plotting the histogram
plt.figure(figsize=(12, 6))
plt.bar(range(len(top_words)), word_percentages, color='skyblue')
plt.xlabel('Words')
plt.ylabel('Percentage')
plt.title('Top 30 Most Used Words in Abstracts')
plt.xticks(range(len(top_words)), top_words, rotation=90)

# Show percentage on top of bars
for i, percentage in enumerate(word_percentages_rounded):
    plt.text(i, percentage + 0.5, f'{percentage}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# %%[markdown]
# Number of Unique Labels

#%%
# Extract labels from strings and flatten them
all_labels = [int(label.strip('[]')) for sublist in train_data['numerical_classification_labels'].str.split() for label in sublist if label.strip('[]')]

# Find the unique labels
unique_labels = np.unique(all_labels)

# Get the total number of unique labels
num_unique_labels = len(unique_labels)

print("Total number of unique labels:", num_unique_labels)
print("Unique labels:")
print(unique_labels)

# %%
# Count the frequency of each label
label_counts = Counter(all_labels)

# Get the top 10 most frequent labels
top_labels = label_counts.most_common(10)

# Extract label values and their frequencies
top_label_values = [label[0] for label in top_labels]
top_label_frequencies = [label[1] for label in top_labels]

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(top_label_values)), top_label_frequencies, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequently Used Labels')
plt.xticks(range(len(top_label_values)), top_label_values)
plt.tight_layout()
plt.show()

# %%
# Initialize an empty dictionary for label mapping
label_mapping = {}

# Iterate over the rows of the DataFrame
for index, row in train_data.iterrows():
    # Extract labels and their numerical encodings
    actual_labels = row['classification_labels']
    encoded_labels = row['numerical_classification_labels']
    
    # Split the labels and encoded values
    actual_labels = actual_labels.strip("[]").split("' '")
    encoded_labels = [int(label) for label in encoded_labels.strip("[]").split()]
    
    # Map each label to its encoded value
    for actual_label, encoded_label in zip(actual_labels, encoded_labels):
        label_mapping[actual_label.strip("'")] = encoded_label

# Print the label mapping dictionary
print(label_mapping)


#%%
# Flatten the lists in the 'classification_labels' column and split them
all_labels = [label.strip("''") for sublist in train_data['classification_labels'] for label in sublist.strip('[]').split("' '")]

# Count the frequency of each label
label_counts = Counter(all_labels)

# Get the top 10 most frequent labels
top_labels = label_counts.most_common(10)

# Extract label values and their frequencies
top_label_values = [label[0] for label in top_labels]
top_label_frequencies = [label[1] for label in top_labels]

# Calculate total number of labels
total_labels = sum(top_label_frequencies)

# Calculate percentages
label_percentages = [(freq / total_labels) * 100 for freq in top_label_frequencies]

# Plotting the bar plot with y-axis in percentage
plt.figure(figsize=(10, 6))
plt.bar(range(len(top_label_values)), label_percentages, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Percentage')
plt.title('Top 10 Most Frequently Used Labels (From classification_labels)')
plt.xticks(range(len(top_label_values)), top_label_values, rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%%
# Combine all abstracts into a single string
all_abstracts = ' '.join(train_data['abstract'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_abstracts)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Abstracts')
plt.show()

#%%
# Convert abstracts to strings, handling any float values
all_abstracts = ' '.join(str(abstract) for abstract in train_data['abstract'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_abstracts)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Abstracts')
plt.show()

#%%
# Filter out float values and calculate length of each abstract
abstract_lengths = train_data['abstract'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

# Plot histogram of abstract lengths
plt.figure(figsize=(8, 6))
plt.hist(abstract_lengths, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Abstract Length')
plt.ylabel('Frequency')
plt.title('Histogram of Abstract Lengths')
plt.xlim(0, 750)  # Set x-axis range from 0 to 750
plt.show()
