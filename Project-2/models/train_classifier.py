import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Load data from a SQLite database and convert it into a DataFrame.
    Parameters
    ----------
    database_filepath (str): The filepath or path to the SQLite database file. 

    Returns
    -------
    X (array-like): The features or independent variables extracted from the database.
    Y (array-like): The target variable or dependent variable extracted from the database.
    category_names (list): The list of category names associated with the target variable.
    """
    # Creating a database engine to connect to the specified SQLite database file
    engine = create_engine('sqlite:///%s'%(database_filepath))
    # Reading data from the SQLite database table
    df = pd.read_sql('SELECT * FROM disaster_category_message', engine)
    # Extracting the 'message' column values as input feature X
    X = df['message'].values
    # Extracting the values of the category columns (columns 5 onward) as target labels Y
    Y = df[df.columns[4:].tolist()].values
    # Extracting the names of the category columns as category names for output labels
    category_names = df.columns[4:]
    return X,Y,category_names


def tokenize(text):
    """
    Split a given text into individual tokens or words.
    Parameters
    ----------
    text (str): The input text to be tokenized.

    Returns
    -------
    clean_tokens (list): A list of tokens generated from the input text
    """
    # Tokenizing the input 'text'
    tokens = word_tokenize(text)
    # Creating a WordNetLemmatizer instance
    lemmatizer = WordNetLemmatizer()
    # Initializing an empty list
    clean_tokens = []
    # Looping through each token for lemmatization and cleaning
    for tok in tokens:
        # Lemmatizing the token, converting to lowercase, and removing leading/trailing spaces
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # Appending the cleaned token to the 'clean_tokens' list
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Build a machine learning model.

    Returns
    -------
    model : The machine learning model that has been built.
    
    """
    # Creating a pipeline for text classification
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Defining parameters for hyperparameter tuning
    parameters = {
        'clf__estimator__max_depth' : [5,None]
    }
    # Creating a model using GridSearchCV
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a machine learning model and returns classification report.
    
    Parameters
    ----------
    model: The trained machine learning model to be evaluated.
    X_test: The input features of the test dataset.
    Y_test: The corresponding true labels of the test dataset.
    category_names: The names of the categories or classes being predicted.

    """
    # Predicting categories
    Y_prediction = model.predict(X_test)
    # Looping through each category and printing classification report
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test[:, i], Y_prediction[:, i]))
        print("="*50)

def save_model(model, model_filepath):
    """
    Save the trained model.

    Parameters
    ----------
    model (object): The trained model object to be saved.
    model_filepath (str): The path and filename for saving the model.

    """
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()