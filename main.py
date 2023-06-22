import openai
import pandas as pd
import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')

# Load your CSV file
data = pd.read_csv('data.csv')

# Initialize your GPT model (make sure you have your OpenAI API key)
openai.api_key = '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'

def generate_prompt(question):
    # This function generates a prompt that includes the question and some relevant context from the CSV data.
    # For simplicity, let's assume that the first row of the CSV file is always relevant.
    context = data.iloc[0].to_dict()
    context_string = "\n".join(f"{key}: {value}" for key, value in context.items())
    return f"{context_string}\n\n{question}"

def answer_generic_question(question):
    original_question = question
    question = question.lower()

    if "total sales from" in question:
        sales_city = re.search(r'total sales from the city (.*)', original_question, re.IGNORECASE)
        sales_state = re.search(r'total sales from the state (.*)', original_question, re.IGNORECASE)
        sales_country= re.search(r'total sales from the country (.*)', original_question, re.IGNORECASE)
        sales_order_num = re.search(r'total sales from the order number (.*)', original_question, re.IGNORECASE)  # New line

        if sales_city:
            city = sales_city.group(1).strip().lower()
            return data.loc[data['CITY'].apply(lambda x: x.lower().strip()) == city, 'SALES'].sum()

        elif sales_state:
            state = sales_state.group(1).strip().lower()
            return data.loc[data['STATE'].apply(lambda x: x.lower().strip()) == state, 'SALES'].sum()

        elif sales_country:
            country = sales_country.group(1).strip().lower()
            return data.loc[data['COUNTRY'].apply(lambda x: x.lower().strip()) == country, 'SALES'].sum()

        elif sales_order_num:  # New block
            order_num = sales_order_num.group(1).strip()
            return data.loc[data['ORDER NUMBER'] == int(order_num), 'SALES'].sum()


    elif "number of orders shipped for the order number" in question:  # Change in the condition
        shipped_order_num = re.search(r'number of orders shipped for the order number (.*)', original_question, re.IGNORECASE)  # New line
        if shipped_order_num:  # New block
            order_num = shipped_order_num.group(1).strip()
            shipped_data = data[(data['ORDER NUMBER'] == int(order_num)) & (data['STATUS'].str.lower() == 'shipped')]
            return shipped_data.shape[0]  # Return the number of rows


    elif "total sales" in question:
        return data['SALES'].sum()

    elif "max sales" in question:
        return data['SALES'].max()

    elif "min sales" in question:
        return data['SALES'].min()

    elif "quantity ordered" in question:
        # Check for specific order number
        order_num_match = re.search(r'total quantity ordered from the order number (.*)', original_question, re.IGNORECASE)
        customer_name_match = re.search(r'total quantity ordered by customer (.*)', original_question, re.IGNORECASE)

        if order_num_match:
            order_num = order_num_match.group(1).strip()
            return data.loc[data['ORDER NUMBER'] == int(order_num), 'QUANTITY ORDERED'].sum()

        elif customer_name_match:
            customer_name = customer_name_match.group(1).strip().lower()
            return data.loc[data['CUSTOMER NAME'].apply(lambda x: x.lower().strip()) == customer_name, 'QUANTITY ORDERED'].sum()

    elif "numbers of orders shipped to"  or "how many orders are shipped to" in question:
        shipped_city_match = re.search(r'number of orders shipped to the city (.*)', original_question, re.IGNORECASE)
        shipped_state_match = re.search(r'number of orders shipped to the state (.*)', original_question, re.IGNORECASE)
        shipped_country_match = re.search(r'number of orders shipped to the country (.*)', original_question, re.IGNORECASE)
            
                    # Filter data to include only rows where STATUS is "Shipped"
        shipped_data = data[data['STATUS'].str.lower() == "shipped"]
        if shipped_city_match and shipped_city_match.group(1).strip().lower() in uni_city:
            return shipped_data['CITY'].str.lower().value_counts().get(shipped_city_match.group(1).strip().lower(), 0)
        elif shipped_state_match and shipped_state_match.group(1).strip().lower() in uni_state:
            return shipped_data['STATE'].str.lower().value_counts().get(shipped_state_match.group(1).strip().lower(), 0)
        elif shipped_country_match and shipped_country_match.group(1).strip().lower() in uni_country:
            return shipped_data['COUNTRY'].str.lower().value_counts().get(shipped_country_match.group(1).strip().lower(), 0)

    else:
        return None

def answer_question(question):
    generic_answer = answer_generic_question(question)
    if generic_answer is not None:
        return generic_answer

    prompt = generate_prompt(question)
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=60)
    answer = response.choices[0].text.strip()

    not_found_patterns = [
        r"couldnt find the answer",
        r"not found",
        r"information not available",
        r"no matching records",
        r"data not found",
        r"is not specified",
        r"unknown",
        r"not given",
        r"not provided"
    ]

    for pattern in not_found_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            input_tokens = tokenize(preprocess_text(question))
            similarity_scores = calculate_similarity_scores(model, column_name_tokens, input_tokens)
            relevant_columns = get_relevant_columns(column_names, similarity_scores, threshold)
            order_number = extract_order_number(input_tokens)

            if order_number is not None:
                relevant_entries = get_relevant_entries(data, relevant_columns, order_number)

                if len(relevant_entries) > 0:
                    return relevant_entries.to_string(index=False)

    return answer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize(input):
    tokens = word_tokenize(input)
    return tokens

def calculate_similarity_scores(model, column_name_tokens, input_tokens):
    similarity_scores = []
    for name_tokens in column_name_tokens:
        name_vector = sum(model.wv[token] for token in name_tokens if token in model.wv) / len(name_tokens)
        input_vector = sum(model.wv[token] for token in input_tokens if token in model.wv) / len(input_tokens)

        if isinstance(name_vector, float):
            continue

        name_vector = np.reshape(name_vector, (1, -1))
        input_vector = np.reshape(input_vector, (1, -1))
        similarity = cosine_similarity(input_vector, name_vector)[0][0]
        similarity_scores.append(similarity)

    return similarity_scores

def get_relevant_columns(column_names, similarity_scores, threshold):
    relevant_columns = [column_names[i] for i, score in enumerate(similarity_scores) if score >= threshold]
    return relevant_columns

def extract_order_number(input_tokens):
    order_number = None
    pattern = r"\b\d+\b"  # regex pattern to match any sequence of digits
    for token in input_tokens:
        match = re.match(pattern, token)
        if match:
            order_number = match.group()
            break
    return order_number

def get_relevant_entries(data, relevant_columns, order_number):
    relevant_data = data[data['ORDER NUMBER'] == int(order_number)][relevant_columns]
    relevant_data.reset_index(drop=True, inplace=True)  # Remove row index
    return relevant_data

# Preprocess the column names
column_names = list(data.columns)
column_name_tokens = [nltk.word_tokenize(re.sub(r'\W', ' ', name.lower())) for name in column_names]

# Train the Word2Vec model
model = Word2Vec(column_name_tokens, min_count=1)

# Set the threshold for similarity scores
threshold = 0.5

data = pd.read_csv('data.csv')
word2vec_model_path = 'word2vec.model'
try:
    model = Word2Vec.load(word2vec_model_path)
except FileNotFoundError:
    model = Word2Vec(column_name_tokens, min_count=1)
    model.save(word2vec_model_path)

# Set the threshold for similarity scores
threshold = 0.5

# Define a function to answer questions using the saved model
def answer_question_with_saved_model(question):
    return answer_question(question, model, column_name_tokens, threshold)
question = input("Enter your question: ")
answer = answer_question(question)
print(answer)