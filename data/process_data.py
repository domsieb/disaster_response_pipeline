import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories into pandas dataframe

    Args:
        messages_filepath: A csv file containing disaster messages
        categories_filepath: A csv file containing disaster categories for each message

    Returns:
        pandas dataframe of merged messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merged_df = messages.merge(right=categories, on="id", how="inner")
    return merged_df


def clean_data(df):
    """Load messages and categories into pandas dataframe

    Args:
        df: pandas dataframe that has merged entries from categories and messages

    Returns:
        cleaned pandas dataframe where the categories are transformed to numeric values
    """
    # Copy dataframe to local dataframe
    df_clean = df
    # Split category into subcategories
    categories = df.categories.str.split(";", expand=True)
    # Label columns according to new label
    categories.columns = categories.iloc[0].str[:-2]
    # Make columns numeric, i.e. remove the label substring from the content
    for label, content in categories.iteritems():
        categories[label] = pd.to_numeric(content.str.replace(f"{label}-", ""))
    # Clean related category to 0/1 - there are outliers with 2s
    categories["related"] = categories["related"].map(lambda x: 1 if x == 2 else x)
    # Drop original category column
    df_clean = df_clean.drop(labels="categories", axis=1)
    # Add categories to dataframe
    df_clean = df_clean.join(categories)

    return df_clean


def save_data(df, database_filename):
    """Load cleaned data frame into an SQL database

    Args:
        df: cleaned pandas dataframe provided by
        database_filename: Path to sql database into which the dataframe is stored.

    Returns:
        None
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("YourTableName", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
