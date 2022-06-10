# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

  '''
  load data from messages and categories datasets

  INPUT:
    messages_filepath: file path for the messages dataset
    categories_filepath: file path for the categories dataset

  OUTPUT:
    a dataframe that has combined data from the two input datasets
  '''

  # load messages dataset
  messages = pd.read_csv(messages_filepath)

  #load categories dataset
  categories = pd.read_csv(categories_filepath)

  # merge datasets
  df = messages.merge(categories, how='left', on = 'id')

  return df


def clean_data(df):

  '''
  clean the data in the dataframe

  INPUT:
    df: dataframe that has the messages and categories data

  OUTPUT:
    cleaned dataframe
  '''

  # create a dataframe of the 36 individual category columns
  categories = df['categories'].str.split(';', expand = True)

  # select the first row of the categories dataframe
  row = categories.iloc[0]

  # use this row to extract a list of new column names for categories

  category_colnames = []
  for i in range(len(row)):
    category_colnames.append(row[i][:-2])

  # rename the columns of `categories`
  categories.columns = category_colnames

  # convert category values to just numbers 0 or 1
  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])

  # drop the original categories column from `df`
  df.drop('categories', axis = 1, inplace = True)

  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], axis=1)

  # drop duplicates and NaN
  df.drop_duplicates(keep=False, inplace=True)
  df = df.dropna()

  return df


def save_data(df, database_filename):

  '''
  save the data into a sql lite database

  INPUT:
    df: dataframe with the cleaned messages + categories data
    database_filename: name of the database 
  '''

  engine = create_engine('sqlite:///' + database_filename)
  df.to_sql('disaster_response', con = engine, if_exists = 'replace', index=False)  


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