import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def load_data(database_filepath):
    """

    Args:
        database_filepath: SQL database

    Returns:
        X: pandas dataframe with features
        Y: pandas dataframe with target values
        category_name: target labels with categories
    """

    database_filepath = "data/DisasterResponse.db"
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("YourTableName", con=engine)

    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    # init stop words and lemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build machine learning model with pipeline and gridsearchcv

    Returns: ML model after grid search

    """

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", RandomForestClassifier()),
        ]
    )

    # specify parameters for grid search
    parameters = {
        "clf__n_estimators": [6,8,10,12,14],
        "tfidf__norm": ["l2", "l1"],
        "tfidf__smooth_idf": [True, False],
        "vect__ngram_range": ((1, 1), (1, 2)),
        "vect__max_df": (0.5, 1.0, 1.5),
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model statistics on test data

    Args:
        model: the actual ml model
        X_test: features of testing set
        Y_test: targets of testing set
        category_names: labels from target

    Returns: None
    """
    Y_pred = model.predict(X_test)

    target_names = category_names
    report = classification_report(
        Y_test, Y_pred, target_names=target_names, zero_division=0
    )
    accuracy = (Y_pred == Y_test).mean()

    print("Classification Report:\n", report)
    print("Accuracy:\n", accuracy)


def save_model(model, model_filepath):
    """
    Saves the ML model as pickle
    Args:
        model: the actual ML model to save
        model_filepath: filepath to the pickle file where the model is supposed to be stored
    Returns: None
    """

    # Save a model into a pickle file
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
