import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function responsible for loading the data
    Parameters
    ----------
    messages_filepath (string): The filepath of the messages file.
    categories_filepath (string): The filepath of the categories file.

    Returns
    -------
    df (pandas DataFrame): The loaded and preprocessed dataset containing 
        the combined information from the messages and categories files.
    """
    # read files and create dataframes
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    # merge both dataframes by id
    df = messages.merge(categories,how = 'left',on = 'id')
    return df

def clean_data(df):
    """
    Perform cleaning and preprocessing operations on the given dataset.
    Parameters
    ----------
    df (pandas DataFrame): The input dataset that requires cleaning.

    Returns
    -------
    df (pandas DataFrame): The cleaned and preprocessed dataset.
    """
    # split categories column
    categories = df['categories'].str.split(';',expand = True)
    # select first row
    row = categories.iloc[1,:]
    # slice the string value up to the second to last character
    category_colnames = row.apply(lambda x: x[0:-2:1]).tolist()
    # rename dataframe
    categories.columns = category_colnames
    # iterate through the category columns to keep only the last character of each string
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # replace 2s by 1s
        categories[column] = categories[column].replace(2,1)
    # drop categories column
    df = df.drop(columns = ['categories'],axis = 1)
    # concat original dataframe and categories data
    df = pd.concat([df,categories],axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Saves the DataFrame as a table in a SQLite database. 
    Parameters
    ----------
    df (pandas DataFrame): The DataFrame containing the data to be saved.
    database_filename (str): The filename or path to the SQLite database
        where the DataFrame will be saved.
    
    """
    engine = create_engine('sqlite:///%s'%(database_filename))
    df.to_sql('disaster_category_message', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()