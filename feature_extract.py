import re
from sentence_splitter import SentenceSplitter
import json
import textstat
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


def sentence_count(text, splitter):
    num_sentence = splitter.split(text=text)
    return len(num_sentence)

def word_count(text):
    words = text.split()
    return len(words)

def flesch_kinkaid_grade(text):
    return textstat.flesch_kincaid_grade(text)

def extract_nouns(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Perform POS tagging
    tagged_words = pos_tag(words)
    # Extract nouns
    nouns = [word for word, pos in tagged_words if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
    # Remove stopwords to refine the results, although not necessary for just nouns
    stop_words = set(stopwords.words('english'))
    filtered_nouns = [noun for noun in nouns if noun.lower() not in stop_words]
    # Return the unique nouns
    unique_nouns = set(filtered_nouns)
    return nouns, unique_nouns

def extract_prepositions(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Perform POS tagging
    tagged_words = pos_tag(words)
    # Count prepositions (tagged as 'IN')
    preposition = [word for word, pos in tagged_words if pos == 'IN']
    return preposition

def extract_numerical_values(text):
    # Regular expression to match numerical values, including integers, decimals, and negative numbers
    num_pattern = r'-?\d+\.?\d*'
    # Find all matches in the text
    numerical_values = re.findall(num_pattern, text)
    # Convert matched values to appropriate numerical types
    numerical_values = [float(num) if '.' in num else int(num) for num in numerical_values]
    return numerical_values

def extract_text_numerical_values(q):
    WORD_ARGS_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    WORD_ARGS_TEENS = {"eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
    WORD_ARGS_TENS = {"twenty": 20, "thirty": 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    WORD_ARGS_21_99 = {f"{tens} {ones}".strip(): t + o for tens, t in WORD_ARGS_TENS.items() for ones, o in WORD_ARGS_ONES.items()}
    WORD_ARGS_FRAC = {"half": 1/2, "one third": 1/3, "a third": 1/3, "two third": 2/3, "one fourth": 1/4, "a fourth": 1/4, "three fourth": 3/4}
    WORD_NUMERALS = {"zero": 0, "ten": 10}
    WORD_NUMERALS.update(WORD_ARGS_ONES)
    WORD_NUMERALS.update(WORD_ARGS_TEENS)
    WORD_NUMERALS.update(WORD_ARGS_TENS)
    WORD_NUMERALS.update(WORD_ARGS_21_99)
    WORD_NUMERALS.update(WORD_ARGS_FRAC)
    # Sort the keys by length in descending order
    sorted_keys = sorted(WORD_NUMERALS.keys(), key=len, reverse=True)
    # Initialize an empty list to collect numerical values
    numerical_values = []
    # Iterate through sorted keys and search for matches
    q = re.sub(r'[^a-z\s]', ' ', q.strip().lower())
    for word in sorted_keys:
        regex = re.compile(r'\b' + re.escape(word) + r'\b')
        matches = re.findall(regex, q)
        if matches:
            numerical_values.extend([WORD_NUMERALS[word]] * len(matches))
            # Remove matched words from the text
            q = re.sub(regex, '', q)
    return numerical_values

def extract_math_operators(text):
    # Define a more comprehensive regular expression pattern for mathematical operators
    pattern = r'[+\-*^><≤≥÷]'
    # Find all matches in the text
    operators = re.findall(pattern, text)
    # get unique operators
    unique_operators = list(set(operators))
    return operators, unique_operators


def main():
    with open("eedi_math_reasoning_data_with_selection_counts.json", "r") as f:
        questions = json.load(f)
    sentence_splitter = SentenceSplitter(language='en') 
    feature_data = []

    for question in questions:
        feature_question = {}
        question_text = f"Question: {question['QuestionStem']}\nOption A: {question['CorrectAnswer']}\nOption B: {question['Student1Answer']}\nOption C: {question['Student2Answer']}\nOption D: {question['Student3Answer']}"
        num_sentences = sentence_count(question_text, sentence_splitter)
        num_words = word_count(question_text)
        flesch_kinkaid = flesch_kinkaid_grade(question_text)
        nouns, unique_nouns = extract_nouns(question_text)
        num_nouns, num_unique_nouns = len(nouns), len(unique_nouns)
        prepositions = extract_prepositions(question_text)
        num_prepositions = len(prepositions)
        numerical_values = extract_numerical_values(question_text)
        num_numerical_values = len(numerical_values)
        text_numerical_values = extract_text_numerical_values(question_text)
        num_text_numerical_values = len(text_numerical_values)
        operators, unique_operators = extract_math_operators(question_text)
        num_operators, num_unique_operators = len(operators), len(unique_operators)
        feature_question["num_sentences"] = num_sentences
        feature_question["num_words"] = num_words
        feature_question["flesch_kinkaid"] = flesch_kinkaid
        feature_question["num_nouns"] = num_nouns
        feature_question["num_unique_nouns"] = num_unique_nouns
        feature_question["num_prepositions"] = num_prepositions
        feature_question["num_numerical_values"] = num_numerical_values
        feature_question["num_text_numerical_values"] = num_text_numerical_values
        feature_question["num_operators"] = num_operators
        feature_question["num_unique_operators"] = num_unique_operators
        feature_question["difficulty"] = question["GroundTruthDifficulty"]
        feature_question["Question_text"] = question_text
        feature_question["ID"] = question["ID"]
        feature_data.append(feature_question)
    
    print(len(feature_data))
    with open("eedi_math_feature_data.json", "w") as f:
        json.dump(feature_data, f, indent=4)
    
if __name__ == "__main__":
    main()

