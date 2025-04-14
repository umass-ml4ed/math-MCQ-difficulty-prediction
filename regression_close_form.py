import json
import numpy as np   
from sklearn.preprocessing import StandardScaler
from utils import draw, initialize_seeds, count_greater_pairs, calculate_r2

def normalize_features(train_X, test_X):
    scaler = StandardScaler()
    # Fit the scaler on the training data
    normalized_train_X = scaler.fit_transform(train_X)
    # Transform the test data using the same scaler
    normalized_test_X = scaler.transform(test_X)
    return normalized_train_X, normalized_test_X

def closed_form_solution(X, y):
    # Adding a column of ones to X for the intercept term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # Closed-form solution: (X^T * X)^-1 * X^T * y
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    return theta

    
if __name__ == "__main__":
    feature_filename = "eedi_math_feature_data.json"
    print("Reading features from:", feature_filename)
    with open(feature_filename, "r") as f:
        questions = json.load(f)
        
    print("Number of questions:", len(questions))
        
    # shuffle the questions
    initialize_seeds(45)
    np.random.shuffle(questions)
    
    # split the data into 5 folds
    fold = 4
    split_point = int((fold / 5) * len(questions))
    questions = questions[split_point:] + questions[:split_point]
    
    all_number_sentences = [question["num_sentences"] for question in questions]
    all_flesch_kinkaid = [question["flesch_kinkaid"] for question in questions]
    all_number_nouns = [question["num_nouns"] for question in questions]
    all_number_unique_nouns = [question["num_unique_nouns"] for question in questions]
    all_number_prepositions = [question["num_prepositions"] for question in questions]
    all_number_numerical_values = [question["num_numerical_values"] for question in questions]
    all_number_text_numerical_values = [question["num_text_numerical_values"] for question in questions]
    all_number_operators = [question["num_operators"] for question in questions]
    all_number_unique_operators = [question["num_unique_operators"] for question in questions]
       
    X = np.array([all_number_sentences, all_flesch_kinkaid, all_number_nouns,
                    all_number_unique_nouns, all_number_prepositions, all_number_numerical_values, 
                    all_number_text_numerical_values, all_number_operators, all_number_unique_operators]).T
    feature_names = ["num_sentences", "flesch_kinkaid", "num_nouns", "num_unique_nouns", "num_prepositions", "num_numerical_values", "num_text_numerical_values", "num_operators", "num_unique_operators"]
        
    print("number of features:", len(feature_names))
    all_difficulties = [question["difficulty"] for question in questions]
    Y = np.array(all_difficulties)
    
    # split and normalize the features
    train_X = X[:int(0.8 * X.shape[0])]
    print("train_X shape:", train_X.shape)
    test_X = X[int(0.8 * X.shape[0]):]
    
    train_X, test_X = normalize_features(train_X, test_X)
    
    train_Y = Y[:int(0.8 * Y.shape[0])]
    print("train_Y shape:", train_Y.shape)
    test_Y = Y[int(0.8 * Y.shape[0]):]
    
    theta = closed_form_solution(train_X, train_Y)
    print("number of parameters:", len(theta))
    # print parameter name and value, the first one should be bias term
    for i in range(len(theta)):
        if i == 0:
            print("bias term", theta[i])
        else:
            print(feature_names[i-1], theta[i])
    print("\n--------------------------------------\n")

    # Predict the difficulty level for the test set
    test_X = np.hstack((np.ones((test_X.shape[0], 1)), test_X))
    predicted_difficulties = test_X @ theta
    
    # Calculate the mean squared error
    mse = np.mean((predicted_difficulties - test_Y) ** 2)
    print("Mean squared error:", mse)
    
    # Calculate the R^2 score
    r2 = calculate_r2(test_Y, np.array(predicted_difficulties))
    print("R^2 score:", r2)
    
    # Calculate MATCH
    gt_pair_coomparisons = np.array(count_greater_pairs(test_Y))
    pred_pair_coomparisons = np.array(count_greater_pairs(predicted_difficulties))
    # calculate the number of times the predicted one matches the ground truth
    correct_pairs_ratio = np.sum(gt_pair_coomparisons == pred_pair_coomparisons) / len(gt_pair_coomparisons)
    print("rato of correct pair comparisons:", correct_pairs_ratio)
