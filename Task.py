#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import unittest
import numpy as np 
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Float
from sqlalchemy.orm import sessionmaker
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column
def load_data_from_csv(file_path):
 """
 Load data from a CSV file into a pandas DataFrame.
 Args:
 file_path (str): Path to the CSV file.
 Returns:
 pandas.DataFrame: DataFrame containing the loaded data.
 """
 print(f"Loading data from {file_path}")
 df = pd.read_csv(file_path)
 print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
 return df
def create_table_from_df(engine, table_name, df):
 """
 Create a database table based on the structure of a DataFrame.
- 22 -
 Args:
 engine: SQLAlchemy engine object.
 table_name (str): Name of the table to be created.
 df (pandas.DataFrame): DataFrame whose structure defines the table schema.
 """
 metadata = MetaData()
 # Create table definition based on DataFrame columns
 columns = [Column('x', Float)]
 for col in df.columns[1:]:
 columns.append(Column(col, Float))
 table = Table(table_name, metadata, *columns)
 metadata.create_all(engine)
 print(f"Table '{table_name}' created with columns: {df.columns}")
def insert_data(session, engine, table_name, df):
 """
 Insert data from a DataFrame into a database table.
 Args:
 session: SQLAlchemy session object.
 engine: SQLAlchemy engine object.
 table_name (str): Name of the table to insert data into.
 df (pandas.DataFrame): DataFrame containing the data to be inserted.
 """
 metadata = MetaData()
- 23 -
 metadata.reflect(bind=engine)
 table = metadata.tables[table_name]
 for index, row in df.iterrows():
 ins = table.insert().values(row.to_dict())
 session.execute(ins)
 session.commit()
 print(f"Inserted {len(df)} rows into table '{table_name}'")
def load_table_to_df(engine, table_name):
 """
 Load data from a database table into a pandas DataFrame.
 Args:
 engine: SQLAlchemy engine object.
 table_name (str): Name of the table to load data from.
 Returns:
 pandas.DataFrame: DataFrame containing the loaded data.
 """
 query = f"SELECT * FROM {table_name}"
 df = pd.read_sql(query, engine)
 return df
def find_best_ideal_functions(training_df, ideal_df):
 """
 Find the best ideal functions for each training function.
- 24 -
 Args:
 training_df (pandas.DataFrame): DataFrame containing training data.
 ideal_df (pandas.DataFrame): DataFrame containing ideal functions.
 Returns:
 dict: A dictionary mapping each training function to its best corresponding ideal function.
 """
 best_ideal_funcs = {}
 for train_col in training_df.columns[1:]: # Skip 'x' column
 min_ssr = float('inf')
 best_func = None
 for ideal_col in ideal_df.columns[1:]: # Skip 'x' column
 ssr = np.sum((training_df[train_col] - ideal_df[ideal_col]) ** 2)
 if ssr < min_ssr:
 min_ssr = ssr
 best_func = ideal_col
 best_ideal_funcs[train_col] = best_func
 return best_ideal_funcs
def approximate_test_data(test_df, ideal_df, best_ideal_funcs):
 """
 Approximate test data using the best ideal functions identified.
 Args:
 test_df (pandas.DataFrame): DataFrame containing test data.
 ideal_df (pandas.DataFrame): DataFrame containing ideal functions.
 best_ideal_funcs (dict): Dictionary mapping each training function to its best ideal function.
- 25 -
 Returns:
 dict: A dictionary containing the residuals for each test data point.
 """
 residuals = {}
 for test_idx, test_row in test_df.iterrows():
 x_val = test_row['x']
 y_test = test_row['y']
 closest_ideal = None
 min_residual = float('inf')
 for ideal_col in best_ideal_funcs.values():
 y_ideal = ideal_df.loc[ideal_df['x'] == x_val, ideal_col].values[0]
 residual = np.abs(y_test - y_ideal)
 if residual < min_residual:
 min_residual = residual
 closest_ideal = ideal_col
 residuals[test_idx] = (x_val, y_test, closest_ideal, min_residual)
 return residuals
class TestFunctions(unittest.TestCase):
 def test_load_data_from_csv(self):
 """Test loading data from CSV file."""
 test_df = load_data_from_csv('test.csv')
 self.assertEqual(len(test_df), 10)
 def test_find_best_ideal_functions(self):
 """Test finding best ideal functions."""
- 26 -
 training_df = pd.DataFrame({'x': range(10), 'y1': np.random.rand(10), 'y2': 
np.random.rand(10)})
 ideal_df = pd.DataFrame({'x': range(10), 'y1': np.random.rand(10), 'y2': 
np.random.rand(10)})
 best_ideal_funcs = find_best_ideal_functions(training_df, ideal_df)
 self.assertEqual(len(best_ideal_funcs), len(training_df.columns) - 1)
def visualize_training_data(training_df):
 """
 Visualize training data using Bokeh.
 Args:
 training_df (pandas.DataFrame): DataFrame containing training data.
 """
 output_file("training_data.html")
 p = figure(title="Training Data", x_axis_label='x', y_axis_label='y', width=800, height=400)
 for col in training_df.columns[1:]:
 p.line(training_df['x'], training_df[col], legend_label=col)
 p.legend.click_policy="hide"
 save(p)
def visualize_test_data(test_df, residuals, ideal_df):
 """
 Visualize test data and residuals using Bokeh.
 Args:
 test_df (pandas.DataFrame): DataFrame containing test data.
 residuals (dict): Dictionary containing residuals for each test data point.
 ideal_df (pandas.DataFrame): DataFrame containing ideal functions.
- 27 -
 """
 output_file("test_data.html")
 p = figure(title="Test Data", x_axis_label='x', y_axis_label='y', width=800, height=400)
 p.circle(test_df['x'], test_df['y'], legend_label='Test Data', color='blue')
 for idx, res in residuals.items():
 p.line([res[0], res[0]], [res[1], ideal_df.loc[ideal_df['x'] == res[0], res[2]].values[0]],
 legend_label=f'Test Data {idx}', color='red')
 p.legend.click_policy = "hide"
 save(p)
class DataVisualizer:
 @staticmethod
 def plot_data(training_df, ideal_df, test_df, residuals, best_ideal_funcs):
 """
 Plot training data, ideal functions, test data, and residuals using Bokeh.
 Args:
 training_df (pandas.DataFrame): DataFrame containing training data.
 ideal_df (pandas.DataFrame): DataFrame containing ideal functions.
 test_df (pandas.DataFrame): DataFrame containing test data.
 residuals (dict): Dictionary containing residuals for each test data point.
 best_ideal_funcs (dict): Dictionary mapping each training function to its best ideal 
function.
 """
 p = figure(title="Training Data vs Ideal Functions")
 for col in training_df.columns[1:]:
 p.line(training_df['x'], training_df[col], legend_label=f"Training {col}", line_width=2)
 for col in best_ideal_funcs.values():
- 28 -
 p.line(ideal_df['x'], ideal_df[col], legend_label=f"Ideal {col}", line_width=2, 
line_dash="dashed")
 p2 = figure(title="Test Data and Residuals", x_range=p.x_range, y_range=p.y_range)
 p2.scatter(test_df['x'], test_df['y'], legend_label="Test Data", color="red")
 for idx, (x, y_test, closest_ideal, residual) in residuals.items():
 p2.line([x, x], [y_test, ideal_df.loc[ideal_df['x'] == x, closest_ideal].values[0]], 
line_width=1, color="black")
 show(column(p, p2))
def main():
 """
 Main function to orchestrate the data loading, analysis, and visualization process.
 """
 # Database setup
 engine = create_engine(r'sqlite:///C:/Users/ECS/Desktop/assignment/database.db')
 Session = sessionmaker(bind=engine)
 session = Session()
 # Load data from tables
 training_df = load_table_to_df(engine, 'training_data')
 ideal_df = load_table_to_df(engine, 'ideal_functions')
 test_df = load_table_to_df(engine, 'test_data')
 # Find the best ideal functions for training data
 best_ideal_funcs = find_best_ideal_functions(training_df, ideal_df)
 print("Best ideal functions identified for each training function:")
- 29 -
 print(best_ideal_funcs)
 # Approximate the test data using the best ideal functions
 residuals = approximate_test_data(test_df, ideal_df, best_ideal_funcs)
 print("Residuals for test data approximation:")
 for idx, res in residuals.items():
 print(f"Test Data Index {idx}: x = {res[0]}, y_test = {res[1]}, closest_ideal = {res[2]}, residual = {res[3]}")
 # Visualize training data
 visualize_training_data(training_df)
 # Visualize test data
 DataVisualizer.plot_data(training_df, ideal_df, test_df, residuals, best_ideal_funcs)
if __name__ == '__main__':
 main()

