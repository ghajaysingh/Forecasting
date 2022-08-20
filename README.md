# Forecasting

# setting the notebook output to be in the center and creating loading symbol
from IPython.display import HTML, display, Markdown, clear_output
from IPython import get_ipython

display(HTML("""
<style>
.output_png img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
 
.loader {
  border: 5px solid #f3f3f3;
  border-radius: 50%;
  border-top: 5px solid teal;
  border-right: 5px solid grey;
  border-bottom: 5px solid maroon;
  border-left: 5px solid tan;
  width: 20px;
  height: 20px;
  -webkit-animation: spin 1s linear infinite;
  animation: spin 1s linear infinite;
  float: left;
}

@-webkit-keyframes spin {
  0% { -webkit-transform: rotate(0deg); }
  100% { -webkit-transform: rotate(360deg); }
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

</style>
"""))

display(Markdown('<span style="color:darkgreen; font-style: italic; font-size: 15px">Prerequisite Code #1 for <b>image and table aesthetics</b> is EXECUTED!</span>'))

The code in the following cell will enable to keep __track of the activities__ performed across the notebook.

import os
import getpass
import platform
import json
import requests
from datetime import datetime
from pytz import timezone 
import time

# get user name
name = getpass.getuser()

if name == 'jovyan':
    name = os.environ['JUPYTERHUB_USER']
    name = name.replace(".",'_')
else:
    pass

# Description.json file have brick name and version

if platform.system() == 'Windows':
    descrp_json_path = '.\\Source\\Description.json'
else:
    descrp_json_path = './Source/Description.json'

with open(descrp_json_path) as f:
    data = json.load(f)

# Brick version
version = data["version"]

# Brick name
brick_name = "_".join(data["filename"].split('_')[1:2])
# Brick language 
language = data["filename"].split('_')[2]

# Language version
py_version = platform.python_version()

flag = 1

# Get current indian time
india_timezone = timezone('Asia/Kolkata')

# Check if URL is accessible; disable functionality if not accessible
disable_tracking = True
try:
    conn_check = requests.post(url = "https://qa.ird.mu-sigma.com/bricks_data",
                               params = {'brick_name':brick_name,
                                         'brick_language':language,
                                         'brick_version':version,
                                         'language_version':py_version,
                                         'user_name': name,
                                         'time': datetime.now(india_timezone).strftime('%d/%m/%Y %H:%M:%S'),
                                         'cell_name': 'connection_check'})

    if conn_check.status_code == requests.codes.ok:
        disable_tracking = False
except:
    disable_tracking = True
    

def track_cell(cell_id, flag_value, err_description = 'Null'):
    global flag, disable_tracking
    
    if disable_tracking: 
        pass
    else:
        ind_time = datetime.now(india_timezone)
        date = ind_time.strftime('%d/%m/%Y %H:%M:%S')
        nb_data = {
            'brick_name':brick_name,
            'brick_language':language,
            'brick_version':version,
            'language_version':py_version,
            'user_name':name,
            'time':date,
            'cell_name':cell_id,
            's_f':flag_value,
            'no_of_times':"1",
            'description':err_description
        } 

        flag = 1
        try:
            url_value = "https://qa.ird.mu-sigma.com/bricks_data"
            # Hit the API
            requests.post(url = url_value, params = nb_data)
        except Exception as err:
            print("connection error: " + str(err))

display(Markdown('<span style="color:darkgreen; font-style: italic; font-size: 15px">Prerequisite Code #2 for <b>tracking cells</b> is EXECUTED!</span>'))

The code in the following cell is a function that will help to __return first `n` items of the iterable__.

from itertools import islice

def take(n, iterable):
    """
    Return first n items of the iterable as a list
    
    Args:
        n (int): Number of parameters to return from the iterable
        iterable (str): The data structure from which the subset will be taken

    Returns:
        list: Subset of the iterable
    """
    return list(islice(iterable, n))

def print_table(show_df,vis = True,flag = 0):
    """
    Function to convert pandas dataframes into plotly table
    
    Args:
        show_df: Dataframe to be converted 
        visible (bool): Indicating if the trace is to be shown on graph
        
    Returns:
        Plotly table object
    
    """
    global fig
    
    if show_df.empty:
        return None
    
    else:
        if show_df.shape[0] <= 5:
            fig_ht = 50*show_df.shape[0]
        elif show_df.shape[0] > 5 and show_df.shape[0] <= 22:
            fig_ht = 20*show_df.shape[0]
        else:
            fig_ht = 500
        
        if flag==0:
            fig = go.Figure(data=[go.Table(header=dict(values=list(show_df.columns),
                                                       fill_color='grey',align='center',
                                                       font=dict(color='white')),
                                           cells=dict(values=[show_df[i] for i in show_df.columns],
                                                      fill_color='rgba(0,191,255,0.5)',
                                                      align='center', font=dict(color='black')), 
                                           visible = vis)])
            fig.update_layout(width=200*len(show_df.columns), height=fig_ht,
                              margin=dict(l=0,r=0,b=0,t=0,pad=0))
            fig.show(config={'displaylogo': False})
        else:
            fig.add_trace(go.Table(header=dict(values=list(show_df.columns),
                                               fill_color='grey', align='center',
                                               font=dict(color='white')),
                                   cells=dict(values=[show_df[i] for i in show_df.columns],
                                              fill_color='rgba(0,191,255,0.5)',
                                              align='center',font=dict(color='black')), 
                                   visible = vis))
            fig.update_layout(width=200*len(show_df.columns), height=fig_ht,
                              margin=dict(l=0,r=0,b=0,t=0,pad=0))
            return fig

display(Markdown('<span style="color:darkgreen; font-style: italic; font-size: 15px">Prerequisite Code #3 is EXECUTED!</span>'))

# Quick Peek for Notebook

This notebook covers:

1.  __Introduction__:
    This section contains the code snippets for installing all the necessary packages, followed by prerequisite functions for aesthetics.
    1.  Installing the necessary packages (Mandatory)
    2.  Prerequisite -Functions (for aesthetics)
    
2.  __Import libraries__:
    This section imports all the necessary libraries for data loading and processing as well as visualizations.
    1.  Importing libraries for data loading and processing
    2.  Importing libraries for charts and visualization
    
3.  __Import Dataset__:
    In this section, we can import the dataset from LOCAL, a URL or a DATABASE.
    1.  Import dataset from LOCAL
    2.  Import dataset from a URL
    3.  Import dataset from a DATABASE
    
4.  __Data  Definition__:    
    In this section, the user specifies whether their data is panel data or time-series data (single panel) 

5. __User Input Section__:
    In this section, the user can select the following:
    1. Target Variable: The variable to be forecasted
    2. Date Time Variable: The column with timestamps
    3. Panel Variable: The column that defines panels in the data (if any)
    4. External Regressors: Other columns of data that could influence the forecast
    
6. __Date Time Conversion__:
    In this section, the date-time variable is converted into the required Date Time format
    
7. __Quick EDA__:
    In this section the user can explore the time-series pattern of both:
    1. Target Variable
    2. External Regressors
    
8. __Data Summary__:
    In this section, the relevant characteristics of the data are displayed in the following sub-section:
    1. Descriptive: User can view summary statistics like mean, standard deviation and IQR of each column in each panel
    2. Missing Values: Panels that contain missing values
    3. Missing Timestamps: Panels that contain missing timestamps
    4. Singularity Check: Columns of data that show no deviation
    5. Column Datatypes: Datatypes of each column
    6. Balanced/Unbalanced: Shows panels that have an unequal number of datapoints
    
9. __Train Validation Split__:
    In this section, the dataset is split into training and validation sets based on the percentage specified by the user
    
10. __Time Series Conversion__:
    Certain models like Holt Winter's and ARIMA accept data only in time series formation. The data is hence converted in this section.
    
11. __Time Series EDA for Manual Parameter Estimation__:
    Should the user choose to manually feed parameters into the model, this section allows him to select particular panels and perform the necessary     EDA required to estimate those parameters:
    1. Plots of time series of train data
    2. Time series decomposition
    3. Spectral Analysis
    4. Stationarity tests: ADF, KPSS, Variance Ratio and Philips Peron test
    5. Autocorrelation and Partial Autocorrelation plots
    
12. __Multivariate Analysis__:
    This section allows the users to deep dive into the nature of regressors using the following tests:
    1. Cointegration Test
    
13. __Forecasting Models__:
    In this section, the user can train, forecast with and visualize the results of the various models available
    1. Univariate Models: Holt Winter's, TBATS, ARIMA, ELS
    2. Multivariate Models (that incorporate regressors): ARIMAX, Prophet, UCM, Linear Regression, MARS, SVR, VAR, Decision Tree, Random Forest,  XGBoost, GAM, Quantile Regression, PLS, Ridge Regression, Lasso Regression, Elastic Net Regression
       
14. __Model Comparison__:
    In this section, the user can compare the results across models to select the best fit for their data
    
15. __Forecast (Ex-Ante)__:
    In this section, the user can load a new dataset and perform forecasting on it, by choosing any of the above models


# Importing libraries

# Importing libraries for data loading & processing 

if __name__ == '__main__':
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Importing libraries</h2></div>'))

import sys
import requests
import warnings
import re
from pathlib import Path
import random
import math
import pickle
import time
from itertools import groupby, product
from statistics import mode

from tqdm.notebook import tqdm

from IPython.display import display, Markdown, HTML, clear_output
import numpy as np
import pandas as pd

from arch.unitroot import ADF, KPSS, VarianceRatio, PhillipsPerron

from scipy.signal import periodogram

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import statsmodels.tsa.holtwinters as hw
from statsmodels.tsa.api import VARMAX, VAR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests, acf, pacf, coint
from statsmodels.formula.api import ols, quantreg

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LarsCV, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from fbprophet import Prophet
from pyearth import Earth
import pygam
from tbats import TBATS, BATS

## ---------------------------------------------------------------------------------------------------------
# Importing libraries for Database connections

# import pymssql
# import psycopg2

## ---------------------------------------------------------------------------------------------------------
# Importing libraries for Visualisation and Interactivity

from matplotlib import pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "notebook"

from termcolor import colored

warnings.filterwarnings('ignore')
random.seed(369)

if __name__ == '__main__':
    clear_output()
    
display(Markdown('<span style="color:darkgreen; font-style: italic; font-size: 15px">LOADED IMPORTANT LIBRARIES!</span>'))

# User Defined Functions 

Following chunks of code contain user defined functions required for pre-processing of data

def coef_plot(df_subset,j=None,val=None, i = 1,l=1,statistic = None,vis = None):
    sub1 = "panel"
    sub2 = "{} of whole data.".format(statistic)
    
    if i>=2:
        show = False
    else:
        show = True
    
    df_subset = df_subset.round(4)
    if j:
        X = df_subset[statistic]
        Y = df_subset[panel_col]
        symmetric=False
        ERR = (df_subset['75%'] - X) if statistic == 'mean' else None
        ERR_minus = (X  - df_subset['25%']) if statistic == 'mean' else None
    else:
        X = df_subset.coef
        Y = df_subset.panelID
        ERR = df_subset.err
        symmetric=True
        ERR_minus = None

    fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name=sub1,error_x=dict(type='data', symmetric=symmetric,array=ERR, arrayminus=ERR_minus,
                                          color='purple', thickness=1.5, width=3),legendgroup=sub1,showlegend=show,visible=vis,
                             marker=dict(color='purple', size=8)),row=1,col=i)
    if j:
        fig.add_trace(go.Scatter(x = [val]*len(df_subset), y = Y, name = sub2,mode='markers+lines', marker=dict(color='LightSeaGreen', size=2),
                                 legendgroup=sub2,showlegend=show,visible=vis),row=1,col=i)
    else:
        fig.update_yaxes(categoryorder = "total ascending")
        fig.add_shape(type="line", x0=val,
                      y0=list(df_subset.loc[df_subset.coef == min(df_subset.coef), 'panelID'])[0],x1=val,
                      y1=list(df_subset.loc[df_subset.coef == max(df_subset.coef), 'panelID'])[0],
                      line=dict(color="LightSeaGreen", width=4, dash="dashdot"),row = 1, col = i)
    if len(panel_ids) < 10:
        coeff_ht = 100*len(panel_ids)
    else:
        coeff_ht = 1000
    fig.update_layout(height=coeff_ht, width= 400*l, showlegend = False)
    return fig

def acf_pacf_plot(data_,vis):
        acf_res, acf_conf = acf(data_[target_var], nlags= inp_lag, alpha=.05)
        pacf_res, pacf_conf = pacf(data_[target_var], nlags= inp_lag, alpha=.05)
        # ACF
        fig.add_trace(go.Bar(x= list(range(inp_lag)), y= acf_res.tolist(), marker_color = 'maroon', width = 0.07,
                             showlegend= False,visible=vis),row = 1,col = 1)
        fig.add_trace(go.Scatter(x=list(range(inp_lag)), y=acf_conf[:, 0] - acf_res,
                                 line=dict(shape = 'spline',width = 0.01,color='lightgray'),showlegend= False,visible=vis),row = 1,col = 1)
        fig.add_trace(go.Scatter(x=list(range(inp_lag)), y=acf_conf[:, 1] - acf_res,line=dict(shape = 'spline',width = 0.01,color='lightgray'),showlegend= False,visible=vis, fill='tonexty'),
                      row = 1,col = 1)
        # PACF
        fig.add_trace(go.Bar(x= list(range(inp_lag)), y= pacf_res.tolist(),marker_color = 'maroon', width = 0.07,
                             showlegend= False,visible=vis),row = 2,col = 1)

        fig.add_trace(go.Scatter(x=list(range(inp_lag)),y=pacf_conf[:, 0] - pacf_res,
                                 line=dict(shape = 'spline',width = 0.01,color='lightgray'),showlegend= False,visible=vis), row =2,col = 1)
        fig.add_trace(go.Scatter(x=list(range(inp_lag)),y=pacf_conf[:, 1] - pacf_res,line=dict(shape = 'spline',width = 0.01,color='lightgray'),
                                 showlegend= False,visible=vis, fill='tonexty'),row = 2,col = 1)

# Import Dataset

Data Loading can be done in 3 ways from different data sources:

* From __`Local`__

The user can provide __absolute__ or __relative path__ to execute the cell below for accessing the data from your __computer__.

* From __`URL`__ 

Execute the cell below to access the data from a __URL__. _Link provided by the user needs to be a __direct link to the CSV__._
Access the set of sample datasets [here](https://vincentarelbundock.github.io/Rdatasets/datasets.html), choose your dataset, right-click on the `CSV` hyperlink, copy the link and paste it in the user input.

* From __`Database`__

Execute the cell below to access the data from a __Database Server__. Change the __Host IP Address and Credentials__ based on your details.
Given below are example snippets to connect to MS SQL database servers. You can connect to other database servers using the same logic as below. 


In this section, user input for data loading and method of loading data is required. 

* `data_load_method` : available options are __Local__, __URL__, and __Database__.

* `data_path/data_url/database information`: Depending on the selected method, path/url is required. If database is selected, based on the type, few credentials are required such as, database name, user id, host id, password, query etc.

########################################################################################
################################## User Input Needed ###################################
########################################################################################
#Specify the method for loading data
data_load_method = "Local" ## Local, URL, Database
#Specify data path
## provide url link in place of folder path in case of URL mode 
data_path = "./Sample_Datasets/Single_Hotel_TS_Train.csv" # options - for panel './Sample_Datasets/Hotel_Panel_Train.csv' 

#Specify database creds in case loading methos is Database
Server = ''
Database_name =  ''
User_name = ''
Password = ''

Following chunk is used to read data using the specified data loading method

display(Markdown('<div><div class="loader"></div><h2> &nbsp; LOADING</h2></div>'))
if data_load_method == "Local":
    # obtaining the file name from the data path provided
    filename = data_path.replace("\\","/").split("/")[-1]

    if ".csv" in filename:
        data = pd.read_csv(data_path)
    elif ".xlsx" in filname:
        data = pd.read_excel(data_path)  ## Defaulted to take sheet 1.Specify sheetnumber with a ',' otherwise
elif data_load_method == "URL":

    # library to send the request and recieve the content from URL
    from requests import get
    # send request to obtain the data from the URL
    data_from_url = get(data_path)
    clear_output()

    # CSV file name obtained from online
    file_name = data_path.split('/')[-1]

    # file that will be created in your local when the data is pulled from the URL
    file_path = Path(os.path.join(os.getcwd(), 'Sample_Datasets', file_name))

    # create the CSV file in your local and write the data pulled in it
    with open(file_path, "w") as my_empty_csv:
        pass

    file_path.write_bytes(data_from_url.content)

    display(Markdown('<div><div class="loader"></div><h2> &nbsp; LOADING the data </h2></div>'))
    # read the data
    data = pd.read_csv(file_path)
elif data_load_method == "Database":
    # code to establish a connection to the server
    conn = pymssql.connect(Server,
                           User_name,
                           Password,
                           Database_name)
    cursor = conn.cursor()

    #code to fetch the dataset from the server
    data = pd.DataFrame(cursor.fetchall())

data = data.replace(r'^\s*$', np.nan, regex=True) # replacing any empty cell of data type string in any column with NaN
data = data.replace([np.inf, -np.inf], np.nan)    # replacing inf with NaN
# descriptive statistics of categorical column(s)
num_cols = data._get_numeric_data().columns.tolist()
cat_cols = list(set(data.columns) - set(num_cols))

clear_output()
display(Markdown('File having __{} rows__ and __{} columns__ loaded.\
The first 10 rows are shown below:'.format(data.shape[0], data.shape[1])))
display(data.head(100))

# Data Understanding

## Quick peek
Following chunk displays the data-type of columns in the dataset

display(Markdown('The __data summary__ is shown below:'.format(filename)))
print('_'*75+'\n')
display(data.info())
original_cols = data.columns
# obtain the list of numerical columns
num_cols = data._get_numeric_data().columns.tolist()
num_cols_vis = ' || '.join(num_cols)
print(colored("Numerical Columns:\nCount:",'magenta',attrs=['bold']),"{}\n{}".format(len(num_cols),num_cols_vis))
# obtain the list of categorical columns
cat_cols = list(set(original_cols) - set(num_cols))
cat_cols_vis = ' || '.join(cat_cols)
print(colored("\nCategorical Columns:\nCount:",'blue',attrs=['bold']),"{}\n{}".format(len(cat_cols),cat_cols_vis))

## Formatting columns

Here we convert every column name to lower-case format and if there is any _white space_ or _dot_ `.` or _comma_ `,` separating words in any column name, it is replaced with an underscore for convenience.

# formatting the columns
original_cols = data.columns
cols_vis = ' || '.join(original_cols)
print(colored("\nColumn Names (Before Formatting):",'magenta',attrs=['bold']),"\n{}".format(cols_vis))
# list of special characters to be handled while formatting column names to avoid any error
special_chars = r'[?|$|#|@|#|%|!|*|(|)|{|}|^|,|-|.|/|>|<|;|:]'
# lower-case and replace any white-space '_' for every column name
data.columns = list(map(lambda x:re.sub(special_chars,r'',x.lower().replace(' ','_').replace("'","").replace("|","_").replace('-','_')), data.columns))
updated_cols = data.columns
cols_vis = ' || '.join(updated_cols)
print(colored("\nColumn Names (After Formatting):",'blue',attrs=['bold']),"\n{}".format(cols_vis))
num_cols = data._get_numeric_data().columns.tolist()
# obtain the list of categorical columns
cat_cols = list(set(data.columns) - set(num_cols))

## Checking and removing duplicates

Presence of duplicate observations can be misleading, this sections helps get rid of such rows in the datasets.

# length of original data
len_data_before = len(data)
display(Markdown('__The length of the original dataframe__ : {}'.format(len_data_before)))
# check for duplicate rows using parameter 'keep' having 3 possible values
# *first (Default) : considers (counts) duplicates except for the first occurrence.
# *last : considers (counts) duplicates except for the last occurrence.
# *False : considers (counts) all duplicates.

# collect duplicate using 'last'
data_duplicate_avoid_one_val = data[data.duplicated(keep='last')]

# collect all the duplicates
data_duplicate_all = data[data.duplicated(keep=False)]
# get unique count of duplicates
len_data_duplicate_avoid_one_val = len(data_duplicate_avoid_one_val)
len_data_duplicate_all = len(data_duplicate_all)
count_unique_val_with_duplicates = len_data_duplicate_all - len_data_duplicate_avoid_one_val
display(Markdown('\n__The number of unique duplicates__ : {}'.format(count_unique_val_with_duplicates)))

if count_unique_val_with_duplicates == 0:
    print(colored("NO DUPLICATES FOUND!",'green',attrs=['bold']))
else:
    print(colored("DUPLICATES ARE SPOTTED!",'red',attrs=['bold']))

##### Deleting duplicate rows

If there are duplicate values, then it is __recommended to delete__ those rows.

# drop rows having missing values across all the columns
if count_unique_val_with_duplicates == 0:
    print(colored('NO DUPLICATE VALUES to remove!','green',attrs=['bold']))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing</h2></div>'))
    print('Number of rows of the original dataframe: {}'.format(data.shape[0]))
    # dropping the rows
    data.drop_duplicates(keep='last',inplace=True)
    clear_output()

    print(colored('DUPLICATE VALUES removed.','red',attrs=['bold']))
    print('\nAfter removing duplicate values, the number of rows in the dataframe is: {}'.format(data.shape[0]))
    display(Markdown('The __data type summary__ is shown below:'.format(file_name)))
    data.info() 

## Changing data type of columns

In this section, data types for selected column(s) can be updated.

__NOTE__ : If your data contains special character such as `$` or `@` that you want to ignore during conversion, then mention it in the list `special_chars` located in the beginning of the code by separating with `|`. Since `|` itself can't be handled like this, we are separately handling it at first.

###############################################################################################################
########################################### User Input Required ###############################################
###############################################################################################################
col_type_cast = ["class"]  ##List of columns to  change eg- ['max_rooms_capacity']
col_what_type = ["str"]  ##int64, float64, str  eg - ['float64' ]

if col_type_cast == []:
    clear_output()
    display(Markdown('__No data type conversion took place!__'))
    pass
else:
    for each in range(len(col_type_cast)):
        data[col_type_cast[each]] = data[col_type_cast[each]].astype(col_what_type[each])
    display(Markdown('data type of column(s) is updated'))

# Missing Value Analysis

## Missing Value on entire data

Following chunk displays the total missing value percentage in the dataset 

# calculate the percentage of total missing values in the data
percent_msng_val = (data.isnull().sum().sum()/(data.shape[0]*data.shape[1]))*100
display(Markdown('Percentage of missing value in entire dataset is: __{}%__'.format(round(percent_msng_val,4))))

## Missing Value on Column-level

The following code is to visualize the missing values (if any) using bar chart.

# calculate the sum
total_msng_val = data.isnull().sum().sort_values(ascending=False)

if sum(total_msng_val.tolist()) == 0:
    print(colored('NO MISSING VALUES to visualize!','green',attrs=['bold']))
else:
    # calculate the percentage
    percent_msng_val = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)
    # generate a table for displaying the information
    missing_data = pd.concat([total_msng_val, percent_msng_val], axis=1, keys=['Total', 'Percentage']).reset_index()
    missing_data.rename(columns = {'index': 'columns'}, inplace=True)
    missing_data_filtered = missing_data[missing_data['Percentage'] > 0]
    if missing_data_filtered.shape[0] == 1:
        display(Markdown("Only __{}__ column has __{}%__ of missing values!".format(missing_data_filtered.iloc[0]['columns'], 
                                                                                     round(missing_data_filtered.iloc[0]['Percentage'],4))))
        display(Markdown('<span style="color:red">Plot generated only if you have at least 2 columns with missing values'))
    else :
        display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating plot</h2></div>'))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=missing_data_filtered['Percentage'], 
                             y=missing_data_filtered['columns'],orientation='h'))
        fig.update_traces(marker_color='maroon')
        fig.update_xaxes(title = "Percentage missing (%)",range=[0, 100])
        fig.update_yaxes(title = "Columns")
        if missing_data_filtered.shape[1] < 5:
            fig_ht = 300
        else:
            fig_ht = 5*missing_data_filtered.shape[1]+300
        fig.update_layout(title ="Percentage of Missing Value for every column",
                          height = fig_ht, width = 900)

        clear_output()
        fig.show(config={'displaylogo': False})

## Missing value treatment

### Drop column(s) with missing values

The cell below accepts user input and drops the specified columns.

########################################################################################
################################## User Input Needed ###################################
########################################################################################
#Flag to remove missing values
drop_column = True   ## False/True
#List of columns to drop
col_to_drop = ['building_type']

if drop_column:
    data.drop(col_to_drop, axis=1, inplace=True)
    cols = data.columns
    # updated list of numerical columns
    num_cols = data._get_numeric_data().columns.tolist()
    # updated list of categorical columns
    cat_cols = list(set(cols) - set(num_cols))
    display(Markdown('__Column(s) dropped__ : {}'.format(col_to_drop)))
    # displaying the updated dataframe
    print('_'*75)
    display(Markdown('__Updated data type summary__ :'))
    data.info()
else:
    display(Markdown('__No columns were dropped due to missing value!__'))

### Drop row(s) with missing values

Following chunk removes row(s) containing missing values

# calculate the sum
total_msng_val = data.isnull().sum()
if sum(total_msng_val.tolist()) == 0:
    print(colored('NO MISSING VALUES to remove!','green',attrs=['bold']))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing</h2></div>'))
    print('Number of rows of the original dataframe: {}'.format(data.shape[0]))
    # dropping the rows
    data.dropna(axis=0, inplace=True)
    clear_output()
    print(colored('MISSING VALUES removed!','green',attrs=['bold']))
    print('\nAfter removing missing values, the number of rows in the dataframe is: {}'.format(data.shape[0]))
    display(Markdown('The __data summary__ is shown below:'))
    display(data.info())

# User input Section

## Data Type Selection

This notebook can handle two different types of data, namely,

* **Non panel Time-Series Data:**  It is a collection of observations(behavior) for a single subject(entity) at different time intervals(generally equally spaced)
* **Panel Time-Series Data (Longitudinal Data):** It is usually called as Cross-sectional Time-series data as it a collection of observations for multiple subjects at multiple instances.

Available options for data type selection are:
* `Panel`
* `Non Panel`

########################################################################################
################################## User Input Needed ###################################
########################################################################################
data_type = "non-panel" # options - "panel"/"non-panel"

display(Markdown("The selected datatype is: **{}**".format(data_type)))

## Frequency Selection

Frequency of the time series is defined by the periodicity of the date-time column.

The frequency will be stored in `date_time_freq` variable.

Few examples of the input format is given below:

| Variable              | Format                   | Meaning            |
| ----------------------| ------------------------ |--------------------|
| date_time_freq        |     `S`                  |    Second Level data      |
| date_time_freq        |     `T`                  |    Minute Level data      |
| date_time_freq        |     `H`                  |    Hour Level data      |
| date_time_freq        |     `D`                  |    Day Level data      |
| date_time_freq        |     `W`                  |    Week Level data     |
| date_time_freq        |     `MS`                 |    Month Start Level data    |
| date_time_freq        |     `M`                  |    Month End Level data    |
| date_time_freq        |     `Y`                  |    Annual Level data     |

########################################################################################
################################## User Input Needed ###################################
########################################################################################
date_time_freq = 'MS' # options - select from above table based on frequency of data

display(Markdown("""The frequency of the data is:  **{}** """.format(date_time_freq)))

## Panel Column Selection (if any)

If the type of data is panel, the variable that divides the data into cross-sections/panels needs to be selected here

if data_type.lower() == 'panel':
    # the list of columns in dataframe
    original_cols = data.columns
    cols_vis = ' || '.join(original_cols)
    print(colored("\nColumn Names:",'grey',attrs=['bold']),"\n{}".format(cols_vis))
    print('_'*75)
    panel_col = "fac_id" # eg - "FAC_ID"  ###### USER INPUT REQUIRED
    display(Markdown("The column which will be considered as the __unique panel identifier__ is: __{}__".format(panel_col)))
    # list down the panels
    panel_ids = list(set(data[panel_col]))
    display(Markdown('__Panel Information:__'))
    panel_shape = []
    for each_panel in panel_ids:
        panel_data = data.groupby(panel_col).get_group(each_panel)
        panel_shape.append(panel_data.shape[0])

    panel_info = pd.DataFrame()
    panel_info['panel_names'] = panel_ids
    panel_info['size'] = panel_shape
    print_table(panel_info)
else:
    display(Markdown('Not applicable for __single panel time series data__!'))
    panel_col = 'Not Applicable'

## Number Of Lags Selection 

The number of lags will be used throughout the notebook while performing computations requiring lag.

__NOTE__: If you have panel level data, check the lowest count of instances across each panel ID, and then select lag.

########################################################################################
################################## User Input Needed ###################################
########################################################################################
inp_lag = 30 # The ideal value tends to be 0.1% of the input time frame with the maximum value not exceeding 30
########################################################################################

if data_type.lower() == 'panel':
    panel_id_count_lowest = min(panel_info['size'])
    if inp_lag > panel_id_count_lowest:
        print(colored('Lag more than the length of data! Please run this cell again.','red'))
    else :
        clear_output()
        display(Markdown("""The number of lags to be used:  **{}** """.format(inp_lag))) 
else :
    display(Markdown("""The number of lags to be used:  **{}** """.format(inp_lag))) 

## Target Variable Selection

In this section, we will be selecting the target variable. The *“target variable”* is the variable whose values are to be modeled and predicted by using other variables (in case of multivariate forecasting).  

For all the time-series forecasting methods that will be discussed in this notebook, we select a **single target variable**.

########################################################################################
################################## User Input Needed ###################################
########################################################################################
target_var = "occupancy"
display(Markdown("The column which will be considered as the __target variable__ is: __{}__".format(target_var)))
print('_'*75)
original_cols = data.columns # the list of columns in dataframe
cols_vis = ' || '.join(original_cols)
print(colored("\nColumn Names:",'grey',attrs=['bold']),"\n{}".format(cols_vis))

##  Variable(s) Selection

Here, we select variables that may have a direct impact on the target/response variable.

########################################################################################
################################## User Input Needed ###################################
########################################################################################
external_variables = ['max_rooms_capacity', 'avgdailyrate', 'percentgovtnights', 'percentbusiness', 'loyalty_pct'] # options - for panel ['rms_avail_qty', 'age', 'revpar', 'mpi', 'slf']
endog_variables = ['occupancy', 'compet_occupancy'] # options - for panel ['occupancy', 'adr']

# NOTE : External and endogenous columns should not overlap. All the columns here should be NUMERICAL.'
# The target variable is an endogenous variable. Make sure to add it to endog_variable list 
########################################################################################

column_names=data.columns
remaining_cols = list(set(column_names) - set(external_variables))
cat_section = []
for i in external_variables:
    if i in cat_cols:
        cat_section.append(i)
if cat_section != []:
    external_variables = [e for e in external_variables if e not in cat_section]
    data_dumm = pd.get_dummies(data[cat_section])
    external_variables.extend(data_dumm.columns)
    data.drop(cat_section, axis=1, inplace=True)
    data = pd.concat([data,data_dumm], axis=1)
    for i in cat_section:
        cat_cols.remove(i)
remaining_cols.remove(target_var)
external_variables_str = " + ".join(external_variables)
formula = target_var + " ~ "+ external_variables_str
clear_output()
display(Markdown("__Selected exogenous variables__: {}".format(external_variables)))
display(Markdown("__Selected endogenous variables__: {}".format(endog_variables)))

## User Input Summary

display(Markdown("""<span style="color:black;font-size: 15px">The selected datatype is: **{}**""".format(data_type)))
display(Markdown("""<span style="color:black;font-size: 15px">The selected frequency is: **{}**""".format(date_time_freq)))
display(Markdown("""<span style="color:black;font-size: 15px">The selected number of lags is: **{}**""".format(inp_lag)))
display(Markdown("""<span style="color:black;font-size: 15px">The selected panel column is **{}**""".format(panel_col)))
display(Markdown("""<span style="color:black;font-size: 15px">Selected target variable is: **{}**""".format(target_var))) 
display(Markdown("""<span style="color:black;font-size: 15px">Selected exogenous variables (for multivariate forecasting) is/are: **{}**""".format(external_variables))) 
display(Markdown("""<span style="color:black;font-size: 15px">Selected endogenous variables (for multivariate forecasting) is/are: **{}**""".format(endog_variables))) 

# Date-Time Conversion <a name = 'date_time_col'> </a>

In this section, we will choose the date-time column and convert that to proper date-time format for future use. This is a very important as only after selecting this we can select other important components of time series, such as `frequency` of the data, `hierarchy` of the data, etc.

For conversion of column, we will use `pandas`'s **to_datetime()** function.

*Sample Format*

|       date-time       | Format                   |
| ----------------------| ------------------------ |
| 01/01/2013 00:00:00   |  "%d/%m/%Y %H:%M:%S"     |
| 01:01:2013 00:00:01   |  "%d:%m:%Y %H:%M:%S"     |
| 01-01-2013 00:00:02   |  "%d-%m-%Y %H:%M:%S"     |
| 01012013 00:00:03     |  "%d%m%Y %H:%M:%S"       |
| 01:01:2013:00:00:04 AM|  "%d:%m:%Y:%H:%M:%S %p"  |
| 01/01-2013 00:00:05 PM|  "%d/%m-%Y %H:%M:%S %p"  |

**NOTE:** The above mentioned function can't handle any date-time before `1677-09-22 00:12:43.145225` and any date-time after `2262-04-11 23:47:16.854775807`. Make sure data does not contain any date-time outside this range.

########################################################################################
################################## User Input Needed ###################################
########################################################################################
date_time_col = 'datetime' # options - for panel "yearmonth"
ts_format = '%Y-%m-%d' # options - for panel %Y%m
is_timestamp = "n" # options - y/n If your dataset has time stamps and you want to retain it along with the date
########################################################################################

if is_timestamp.lower() == 'y':
    data[date_time_col] = pd.to_datetime(data[date_time_col], format=ts_format)
elif is_timestamp.lower() == 'n':
    data[date_time_col] = pd.to_datetime(data[date_time_col]).dt.date


# ts_format_final = '-'.join(ts_format[i:i+2] for i in range(0, len(ts_format), 2))
# data[date_time_col] = data[date_time_col].map(lambda x: x.strftime(ts_format_final))

data = data.sort_values(date_time_col)

if date_time_col in cat_cols:
    cat_cols.remove(date_time_col)

display(Markdown("The selected date-time column is: __{}__".format(date_time_col)))
display(Markdown('Datetime column converted!'))
if date_time_col in num_cols:
    num_cols.remove(date_time_col)

# Time Series Plots

Following code chunk displays the time-series plots for all the columns in the dataset

########################################################################################
################################## User Input Needed ###################################
########################################################################################
#Specify list of panel ids
selected_panels = ["H3700C06"]
# generating the time series plot for every numerical columns across panels
if data_type.lower() == 'panel':
    if len(num_cols) <= 4:
        row_cnt = 1
        col_cnt = len(num_cols)
    else:
        col_cnt = 4
        row_cnt = len(num_cols)//4+1
        if len(num_cols)%4 == 0:
            row_cnt = len(num_cols)//4

    fig = make_subplots(rows=row_cnt, cols=col_cnt, subplot_titles=num_cols)

    # calculating the number of rows and columns
    r, c = list(range(1,(len(num_cols)//4+2))), list(range(1,5))
    rc_pair = list(product(r, c))

    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the plots</h2></div>'))

    # performing the operation on panel level
    for p_id,each_panel in enumerate(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_panel)
        panel_data = panel_data.sort_values(date_time_col)
        vis = True if each_panel == selected_panels[0] else False

        # generating the plots for every numerical columns
        for c_id,each_col in enumerate(num_cols):
            fig_r = rc_pair[c_id][0]
            fig_c = rc_pair[c_id][1]

            if each_col == target_var:
                lin_col = 'firebrick'
            else:
                lin_col = 'royalblue'
            fig.add_trace(go.Scatter(x=panel_data[date_time_col], y=panel_data[each_col],
                                     line=dict(color=lin_col, width=1), visible=vis), 
                          row=fig_r, col=fig_c)

    panel_dict_list = []
    # creating the drop-down option
    for each_panel in selected_panels:
        vis_check = [[True]*len(num_cols) if i==each_panel else [False]*len(num_cols) for i in selected_panels]
        vis_check_flat = [i for sublist in vis_check for i in sublist]
        panel_dict_list.append(dict(args = [{"visible": vis_check_flat},
                                            {"title": "Observation of each columns in panel: {}".format(each_panel)}],
                                    label=each_panel, method="update"))

    # Add dropdown by opening the option on horizontal direction
    fig.update_layout(updatemenus=[dict(buttons=list(panel_dict_list),
                                        direction="right",
                                        x=0, xanchor="left", y=1.15, yanchor="top")],
                     showlegend=False, title_x=0.5)

else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the plots</h2></div>'))

    row_cnt = len(num_cols)//2+1
    if len(num_cols)%2 == 0:
        row_cnt = len(num_cols)//2

    fig = make_subplots(rows=row_cnt, cols=2, subplot_titles=num_cols)

    # calculating the number of rows and columns
    r, c = list(range(1,(len(num_cols)//2+2))), list(range(1,3))
    rc_pair = list(product(r, c))

    for c_id,each_col in enumerate(num_cols):
        fig_r = rc_pair[c_id][0]
        fig_c = rc_pair[c_id][1]

        if each_col == target_var:
            lin_col = 'firebrick'
        else:
            lin_col = 'royalblue'
        fig.add_trace(go.Scatter(x=data[date_time_col], y=data[each_col],
                                 line=dict(color=lin_col, width=1)),
                      row=fig_r, col=fig_c)
#         fig.update_xaxes(tickangle=90)
if len(num_cols) <= 4:
    fig_ht = 550
elif len(num_cols) > 4 and len(num_cols) <= 20:
    fig_ht = 1000
else:
    fig_ht = 1500
fig.update_layout(width=900, height=fig_ht, showlegend=False, title_x=0.5)
clear_output()
fig.show(config={'displaylogo': False})

# Data Summary

## Data Summary of Numeric Columns

# descriptive and numerical statistics of numerical column(s)
if num_cols == []:
    display(Markdown('__NO NUMERICAL COLUMNS AVAILABLE!__'))
else:
    num_desc = data[num_cols].describe().T
    num_desc.insert(loc=5, column='IQR', value=(num_desc['75%']-num_desc['25%']))
    num_desc.drop(['25%','50%','75%'], axis=1 ,inplace = True)
    num_desc['skewness'] = data[num_cols].skew()
    num_desc['kurtosis'] = data[num_cols].kurt()
    num_desc.insert(loc=0, column='columns', value=num_desc.index)
    num_desc.insert(loc=3, column='median', value=data[num_cols].median())    
    print_table(num_desc.round(4))

## Data Summary of Categorical Columns

if cat_cols == []:
    print('NO CATEGORICAL COLUMNS!')
else:
    cat_summ = pd.DataFrame()
    for col in cat_cols:
        cat_col_summary = pd.DataFrame(data[col].value_counts(dropna=False)).reset_index()
        cat_col_summary.columns = ['Category','Frequency']
        cat_col_summary['Percentage(%)'] = (cat_col_summary["Frequency"]/cat_col_summary["Frequency"].sum())*100
        cat_col_summary['Variable'] = col
        cat_summ = cat_summ.append(cat_col_summary)
    print_table(cat_summ[['Variable','Category','Frequency','Percentage(%)']].round(4))

# Time Series Plots (Panel Data)

## Panel-wise Data Summary

Following code chunks displays numeric data summary for all specified panels from the user

reqrd_cols = num_cols.copy()
if(data_type.lower()=='panel'):
    reqrd_cols.remove(date_time_col) if date_time_col in reqrd_cols else reqrd_cols
    fig  = go.Figure()
    dict_stats = {'Mean':data.groupby(panel_col).mean().round(3).reset_index(),
                  'Standard Deviation': data.groupby(panel_col).std().round(3).reset_index(),
                  'Minimum': data.groupby(panel_col).min()[reqrd_cols].round(3).reset_index(),
                  'Maximum': data.groupby(panel_col).max()[reqrd_cols].round(3).reset_index(),
                  'Distinct count': data.groupby(panel_col).count().round(3).reset_index()}

    for i in dict_stats.keys():
        visible =  True if i == list(dict_stats.keys())[0] else False  
        fig =  print_table(dict_stats[i],vis = visible,flag = 1)

    tab_dict_list = []
    for each_col in dict_stats.keys():
        vis_check = [True if i==each_col else False for i in dict_stats.keys()]

        tab_dict_list.append(dict(args = [{"visible": vis_check}],
                                  label=each_col, method="update"))

        fig.update_layout(updatemenus=[dict(buttons=list(tab_dict_list),
                                            direction="right",x=0, xanchor="left", 
                                            y=1.25, yanchor="top")],
                          showlegend=False)

    fig.show(config={'displaylogo': False})
    display(Markdown("""<span style="color:red;
                    font-size: 15px"> Note: Distinct count value of 1 indicates that the variable is singular. Such variable(s) should not be used as external regressor(s) in multivariate analysis and should be dropped out instead.</span>"""))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

__Dot and whiskers plot__

if data_type.lower()=='panel':
    statistical_method_list = ['mean','std']
    title_list = []

    if date_time_col in num_cols:
        num_cols.remove(date_time_col)

    df_grouped = data.groupby(panel_col)
    df_grouped_stats = df_grouped.describe()

    fig = make_subplots(rows=1, cols=len(num_cols), subplot_titles=tuple(num_cols))
    for k,statistic in enumerate(tqdm(statistical_method_list)):
        if(statistic=='mean'):
            naive_coef = data.mean()
            visible = True
        else:
            naive_coef = data.std()
            visible = False
        for j in num_cols:
            df_subset = df_grouped_stats[j].reset_index()                       
            df_subset = df_subset.sort_values(by=[statistic])
            i = num_cols.index(j)+1
            fig = coef_plot(df_subset,j,naive_coef[j],i,len(num_cols),statistic,visible)

    all_vis = [True]*len(statistical_method_list)
    panel_dict_list = []

    # creating the drop-down option
    for sts in statistical_method_list:
        vis_check = [[True]*len(num_cols)*2 if i==sts else [False]*len(num_cols)*2 for i in statistical_method_list]
        vis_check_flat = [i for sublist in vis_check for i in sublist]

        panel_dict_list.append(dict(args = [{"visible": vis_check_flat},
                                            {"title":  "Panelwise statistics:  {}".format(sts)}],
                                    label=sts, method="update"))

    if len(panel_ids) < 10:
        coeff_ht = 100*len(panel_ids)
    else:
        coeff_ht = 1000
    fig.update_layout(height=coeff_ht, width=400*len(num_cols),
                      showlegend = True,legend_orientation="h")
    # Add dropdown by opening the option on horizontal direction
    fig.update_layout(updatemenus=[dict(buttons=list(panel_dict_list),
                                        direction="right",
                                        x=0, xanchor="left", y=1.25, yanchor="top")],
                     showlegend=True)
    clear_output()
    fig.show(config={'displaylogo': False})
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

# Dropping Unbalanced Panels

Identification of missing timestamps is important while doing a time-series analysis as they can lead to misinterpretation of any kind of forecasting model. 

**Unbalanced** panels (panels having missing timestamps) are identified and then dropped, thereby making the dataset **balanced**

*NOTE:* This section is **applicable only if** the dataset type is **Panel data**

## Check whether the panels are balanced
if data_type.lower() == 'panel':
    panel_count = data.groupby(panel_col).size()
    if min(panel_count) == max(panel_count):
        display(Markdown("__The data has balanced panel!__"))
    else:
        display(Markdown("__Data has unbalanced panels. Please correct this in the following sections.__"))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

## Identifying missing timestamp(s) <a name= 'missing_ts'> </a>


We would identify panels in the dataset that do not have data for all the timestamps


## Creating a padded time-series with all dates
all_timestamps = pd.date_range(start= data[date_time_col].min(),
                               end= data[date_time_col].max(), freq= date_time_freq)
if data_type.lower() == 'panel':
    status = dict()
    missing_ts_dict = dict()

    for each_id in selected_panels:
        panel_data = data.groupby(panel_col).get_group(each_id)
        if panel_data.shape[0] == mode(panel_shape):
            missing_ts_status = 'No'
        else:
            missing_ts_status = 'Yes'

        missing_timestamp = list(set(all_timestamps.astype(str)).difference(set(panel_data[date_time_col].astype(str))))
        count_ts = len(missing_timestamp)
        status.update({each_id: {'is_missing': missing_ts_status,
                                'missing_tp': missing_timestamp, 'missing_count': count_ts}})

    is_missing_list = []
    for primary_key, inner_key in status.items():
        is_missing_list.append(inner_key['is_missing'])

    if 'Yes' in is_missing_list:
        print('\033[35;1mPanels with missing timestamps are as follows: \033[0m')
        for f_id, f_info in status.items():
            if f_info['is_missing'] == 'Yes':
                missing_ts_dict.update({f_id:f_info})
                print('\033[34;1m-----\033[0m'*10)
                print(panel_col, f_id)
        display(Markdown('__The above panels will be removed form the analysis!__'))
    else:
        print('\033[35;1mData does not contain missing timestamps. \033[0m')
        missing_timestamp = []
        is_missing_list = []

# Checking for the missing time-stamps
else:
    if data.shape[0] == len(set(all_timestamps)):
        missing_timestamp = []
        is_missing_list = []
        print('\033[35;1mData does not contain missing timestamps. \033[0m')
    else:
        display(Markdown("__Data contains missing timestamps!__"))
        is_missing_list = ['Yes']

        missing_timestamp = list(set(all_timestamps.astype(str)).difference(set(data[date_time_col].astype(str))))

        display(Markdown("_The missing timestamps shown below._"))
        print_table(pd.DataFrame(missing_timestamp, columns = [date_time_col]))

## Dropping panels with missing timestamp(s)

Any panel(s) not containing all required timestamps are dropped in the following chunk

if data_type.lower()=='panel':
    if len(missing_ts_dict) == len(selected_panels):
        print(colored('All the panels have atleast 1 missing timestamp. \
                      Hence, not dropping any panels, but result will be compromised.','red'))
    else:
        if any(element == 'Yes' for element in is_missing_list):
            #missing_ts_treat = input("Would you like to drop panels with missing stamps? (y / yes)")
            missing_ts_treat = 'y'

            if missing_ts_treat.lower() in ['yes','y']:
                for p_id in missing_ts_dict.keys():   # Removing the panels with missing timestamps: 
                    panel_ids.remove(p_id)
                    if p_id in selected_panels:
                        selected_panels.remove(p_id)
                        pid = random.choice(selected_panels) if pid==p_id else pid ## picking a different pid value if it is a panel id that is meant to be dropped (edit as required)
                display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px"><b>{}</b> Panel(s) with missing timestamp removed!</span>'.format(len(missing_ts_dict))))
            else:
                display(Markdown("__Notebook does not support imputation. Impute data outside the notebook.__"))
        else:
             print('\033[35;1mThis section is not applicable as data doesnt have missing timestamps. \033[0m')
else:
    print('\033[35;1mThis section is not applicable for non-panel data. \033[0m')

# Training-Validation Split

One of the very common issues while developing Machine Learning models is [overfitting](https://en.wikipedia.org/wiki/Overfitting). In order to identify this issue, one needs to validate the Machine Learning model that was build on the observed/training set, on another set of data called the validation set

To do this, we split the data into two parts:
* **Training** - This dataset is used to train our ML model.
* **Validation** - We use this dataset to validate our model. If our expectations are met, we use this to predict/forecast. Otherwise, we take input from this score and try to optimize the parameters in order to improve the performance of our model on the validation set.

Commonly, the training set contains around 70% to 80% of the observations and the validation set contains the remaining 20% - 30%. You can choose the training-validation split in the code chunk below.

![](./Images/TrainTest.png)

########################################################################################
################################## User Input Needed ###################################
########################################################################################
split_percentage = 0.75 # enter values in the range [0.7,0.8]
########################################################################################
unique_timestamp = list(set(data[date_time_col]))
unique_timestamp.sort()
split_condition = unique_timestamp[math.ceil(len(unique_timestamp) * split_percentage)]

series = data.copy()                   ## Creating a copy of the original dataset for future modification
target_series = series[target_var]     ## storing the target series here

## creating training and validation data for the model
train = data[data[date_time_col] <= split_condition]
valid = data[data[date_time_col] >= split_condition] 

clear_output()
display(Markdown("__The distribution of train-validation data accross the observed time is shown below:__"))
display(Markdown('Train-Validation __Split__ Percentage : {}%'.format(split_percentage*100)))
display(Markdown('__Total__ Observations       : {}'.format(len(target_series))))
display(Markdown('__Train__ Observations       : {}'.format(len(train))))
display(Markdown('__Validation__ Observations  : {}'.format(len(valid))))

display(Markdown('***'))
display(Markdown('Beginning timestamp of __train__ data  : {}'.format(min(train[date_time_col]))))
display(Markdown('Ending timestamp of __train__ data     : {}'.format(max(train[date_time_col]))))
display(Markdown('Beginning timestamp of __validation__ data  : {}'.format(min(valid[date_time_col]))))
display(Markdown('Ending timestamp of __validation__ data     : {}'.format(max(valid[date_time_col]))))

# Visulisation
# Creating a plot function to show training-validation plot for non-panel data
if data_type.lower() == 'non-panel':
    fig = go.Figure()                                                   ## Setting the figure and adding trace
    fig.add_trace(go.Scatter(x= train[date_time_col], y= train[target_var], name= "Training",
                             line_color= 'maroon', opacity= 0.8))

    fig.add_trace(go.Scatter(x= valid[date_time_col], y= valid[target_var], name= "Validation",
                             line_color= 'green', opacity=0.8))

    min_ts = min(train[date_time_col])                                           ## Setting the range for the x-axis (time)
    max_ts = max(valid[date_time_col])

    fig.update_layout(xaxis_range=[min_ts, max_ts], title_text="Training-Validation Split",
                      xaxis_title=date_time_col, yaxis_title=target_var, legend_orientation="h")           
    fig.show(config={'displaylogo': False})

# Univariate EDA

## Time-Series Decomposition

The decomposition of time series data is a statistical task that de-constructs a time series into several components to define its characteristics. Usually, a time-series data contains 4 types of pattern:

* __Trend (T)__ : The general change (long-term upward or downward pattern) in the level of the data over a duration longer than a year.
* __Seasonal (S)__ : The regular wavelike fluctuations of constant length, repeating themselves within each 12-month period year after year.
* __Cyclical (C)__ : The _quasi-regular_ (4 phases) wavelike fluctuations - from peak (prosperity) to contractions (recession) to trough (depression) to expansion (recovery) - around the long-term trend, lasting longer than a year.
* __Irregular (R)__ : The short-duration and non-repeating random variations of the data that exist after taking into account the unforeseen events such as strikes or natural disaster.

However, when we decompose the time-series into components, we combine the trend and cycle component into one time-series component to get a __trend-cycle__ component, also known as __Trend__. Thus, time series can be considered of comprising 3 major components: 

* __Trend-cycle__ or __Trend__ component
* __Seasonal__ component
* __Remainder__ component (containing noise)

For decomposition, we need to define the __type__ of the observed series, which is stored in `series_type` variable. It can either be __additive__ or __multiplicative__.
* __Additive__ : The seasonal, cyclical and random variations are absolute deviations from the trend. It is _linear_ where seasonality changes over time are consistently made by having almost the _same frequency_ (width of cycles) and _amplitude_ (height of cycles).

$$\begin{gather*}
y(t) = T_{t}+S_{t}+C_{t}+R_{t}
\end{gather*}$$

* __Multiplicative__ : The seasonal, cyclical and random variations are relative (percentage) deviations from the trend. It is nonlinear, such as quadratic or exponential and non-linear seasonality has an increasing or decreasing frequency and/or amplitude over time.

$$\begin{gather*}
y(t) = T_{t}*S_{t}*C_{t}*R_{t}
\end{gather*}$$

Following code chunk display the time-series decomposition plot based on choice of decomposition

########################################################################################
################################## User Input Needed ###################################
########################################################################################
series_type = "a" # options 'a' for additive / 'm' for multiplicative
enter_freq = 12 # options month- 12,quarter-4,week-52,day-365,hour-3600,minute-60
########################################################################################

decomp_data_cols =  ['Observed', 'Trend', 'Seasonal' ,'Residual']
fig = make_subplots(rows=4, cols=1, subplot_titles=tuple(decomp_data_cols))
if data_type.lower() == 'panel':

    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the plots</h2></div>'))

    for idx,each_id in enumerate(tqdm(selected_panels)):
        panel_data = data.groupby(panel_col).get_group(each_id)

        if panel_data.shape[0] < 2*(enter_freq):
            clear_output()
            print(colored("No plot can be generated since the number of observations are less than twice the integer frequency ",'red'))
            continue
        else:
            # Decomposing the time-series
            decomp_res = seasonal_decompose(panel_data[target_var], model=series_type, 
                                            filt=None, freq=enter_freq)
            decomp_data = pd.DataFrame({'Observed': decomp_res.observed, 'Trend':decomp_res.trend,
                                      'Seasonal':decomp_res.seasonal, 'Residual':decomp_res.resid})

            decomp_data_cols = decomp_data.columns
            decomp_data[date_time_col] = panel_data[date_time_col]
            visible = True if each_id == selected_panels[0] else False
            for i,each_col in enumerate(decomp_data_cols):
                j= i+1
                fig.add_trace(go.Scatter(x= decomp_data[date_time_col],y= decomp_data[each_col],
                                         mode = 'lines', name = 'value', opacity= 0.8, showlegend= False,
                                         visible = visible, legendgroup='value')
                              ,row = j, col = 1)

            tab_dict_list = []
            display(Markdown('__Generating results in tabs...__'))
            for each_id in tqdm(selected_panels):
                vis_check = [[True]*len(decomp_data_cols) if i==each_id else [False]*len(decomp_data_cols) for i in selected_panels]
                vis_check_flat = [i for sublist in vis_check for i in sublist]
                tab_dict_list.append(dict(args=[{"visible": vis_check_flat},
                                                {"title": "Decomposition plots for panel: {}".format(each_id)}],
                                          label=each_id, method="update"))
                fig.update_layout(updatemenus=[dict(buttons=list(tab_dict_list),
                                                    direction="right", x=0, xanchor="left", y=1.11, yanchor="top")],
                                  showlegend=False, title_x=0.5)

            fig.update_layout(height=1000, hovermode='x unified')
            clear_output()
            display(Markdown('Selected __series type__ : {}'.format(series_type)))
            display(Markdown('__Decomposition plot for `{}` with periodicity `{}` generated!__'.format(target_var, enter_freq)))
            fig.show(config={'displaylogo': False})
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the plots</h2></div>'))

    decomp_res = seasonal_decompose(data[target_var], model=series_type, 
                                    filt=None, freq=enter_freq)
    decomp_data = pd.DataFrame({'Observed': decomp_res.observed, 'Trend':decomp_res.trend,
                              'Seasonal':decomp_res.seasonal, 'Residual':decomp_res.resid})
    decomp_data['Residual'].fillna(0, inplace=True)
    decomp_data_cols = decomp_data.columns
    for i,each_col in enumerate(decomp_data_cols):
        j= i+1
        fig.add_trace(go.Scatter(x= data[date_time_col],y= decomp_data[each_col],
                                     mode = 'lines', name = 'value', opacity= 0.8, showlegend= False,
                                     visible = True, legendgroup='value')
                          ,row = j, col = 1)

    fig.update_layout(height=1000, hovermode='x unified')
    clear_output()
    display(Markdown('Selected __series type__ : {}'.format(series_type)))
    display(Markdown('__Decomposition plot for `{}` with periodicity `{}` generated!__'.format(target_var, enter_freq)))
    fig.show(config={'displaylogo': False})

## Stationary Analysis

For a time-series to have the [stationary](https://en.wikipedia.org/wiki/Stationary_process) property,
the series should have three basic properties:

* The mean of the series should NOT be a function of time and should be a constant.
* The variance of the series should NOT be a function of time. This property is known as [homoscedasticity](https://www.statisticssolutions.com/homoscedasticity/).
* The [covariance](https://www.investopedia.com/terms/c/covariance.asp) of the $i^{th}$ term and the $(i+k)^{th}$ term (`k` being the time lag) should NOT be a function of time.

There are 3 different types of stationarity present in time-series analysis:

* **Strict Stationary:**
A strict stationary series satisfies the mathematical definition of a stationary process. For a strict stationary series, the __mean, variance and covariance are not the function of time__. The aim is to convert a non-stationary series into a strict stationary series for making predictions.

* **Trend Stationary:**
A __unit root__ is a feature of some stochastic/random processes (such as random walks). A series that has __NO unit root__ but exhibits a trend is referred to as a trend stationary series. __Once the trend is removed, the resulting series will be strict stationary__.

* **Difference Stationary:**
A time series that can be made strict stationary by differencing is known as difference stationary.

There are several techniques to check whether a time-series is stationary or not such as:

* **Look at Plots:** You can review a time series plot of your data and visually check if there are any obvious trends or seasonality. However, if a time-series has complex stationary patterns, it is not recommended to analyze that using this approach.
* **Summary Statistics:** You can review the summary statistics for your data for seasons or random partitions and check for obvious or significant differences.
* **Statistical Tests:** You can use statistical tests to check if the expectations of stationarity are met or have been violated.

In this notebook, we will be discussing the most popular **statistical tests** to identify the presence of stationarity in the data.


### Parametric tests

##### ADF (Augmented dickey-Fuller) Test

The __[Augmented Dickey-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)__ is a type of statistical test called a __unit root test__. The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.

ADF uses an autoregressive model and optimizes an information criterion across multiple different lag values.

* __Null Hypothesis ($H_0$)__ : The series contains __unit root__, and hence it is non-stationary. It has some time dependent structure.
* __Alternative Hypothesis ($H_1$)__ : The series is weakly stationary. It does not have time-dependent structure.

The function __`ADF`__ has the `trend` parameter with the following 4 options: 
* `nc` - `N`o `C`onstant trend
* `c` - `C`onstant trend (default) 
* `ct` - `C`onstant and linear `T`ime trend
* `ctt` - `C`onstant and linear `T`ime and quadratic `T`ime trends

`lags` is the number of lags to use in the ADF regression. If omitted or None, `method` such as __AIC, BIC__ or __t-stat__ is used to automatically select the lag length.

__HOW TO INTERPRET__ : 
* If the __p-value__ obtained from the test is __less than the significance level of 0.05__, then we fail to accept the $H_0$ i.e., __the series is stationary__.
* If the test statistic is __greater__ than the critical value (for levels of 10%, 5% and 1%), then we accept the $H_0$ i.e., __the series is non-stationary__, and needs to be differenced.

Following code chunk performs adf test on the data

if data_type.lower() == 'panel':
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the result</h2></div>'))

    df_adf = pd.DataFrame()
    for each_id in tqdm(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_id)

        # performing ADF test
        adf_test = ADF(panel_data[panel_data[panel_col] == each_id][target_var])
        
        # assigning whether the null hypothesis is accepted
        stationary_check = 'YES' if adf_test.pvalue < 0.05 else 'NO'

        # storing the result in a dataset
        df_adf_output = pd.DataFrame([round(adf_test.stat,4), round(adf_test.pvalue,4), 
                                      adf_test.lags, stationary_check],
                                    index=['Test Statistic','p-value','Number of Lags Used',
                                          'Stationary']).reset_index()
        df_adf_output.columns = (['Panel','value'])
        df_adf_output['index'] = each_id

        for k,v in adf_test.critical_values.items():
            df_adf_output.loc[df_adf_output.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),each_id])

        df_adf_output = df_adf_output.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
        df_adf_output.index = [x for x in df_adf_output.index]

        df_adf_output.insert(loc=0, column='Panels', value=df_adf_output.index)
        df_adf = df_adf.append(df_adf_output)
else:
    adf_test = ADF(data[target_var])
    
    # assigning whether the null hypothesis is accepted
    stationary_check = 'YES' if adf_test.pvalue < 0.05 else 'NO'

    # storing the result in a dataset
    df_adf_output = pd.DataFrame([round(adf_test.stat,4), round(adf_test.pvalue,4), 
                                  adf_test.lags, stationary_check],
                                index=['Test Statistic','p-value','Number of Lags Used',
                                      'Stationary']).reset_index()
    df_adf_output.columns = (['Panel','value'])
    df_adf_output['index'] = 'NA'

    for k,v in adf_test.critical_values.items():
        df_adf_output.loc[df_adf_output.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),'NA'])

    df_adf_output = df_adf_output.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
    df_adf_output.index = [x for x in df_adf_output.index]
    
    df_adf_output.insert(loc=0, column='Panels', value=df_adf_output.index)
    df_adf = df_adf_output
clear_output()
display(Markdown('__Results of Augmented Dickey-Fuller Test for `{}`:__'.format(target_var)))
if df_adf.shape[0] == 1:
    display(df_adf)
else:
    print_table(df_adf.round(4))

##### KPSS (Kwiatkowski–Phillips–Schmidt–Shin) Test

The __Kwiatkowski–Phillips–Schmidt–Shin test__ is used to determine whether differencing is required on the data. 

* __Null Hypothesis ($H_0$)__ : The series is weakly stationary.
* __Alternative Hypothesis ($H_1$)__ : The series contains __unit root__, and hence it is non-stationary.

Note that the $H_0$ and $H_1$ for the KPSS test are opposite to that of the ADF test, which often creates confusion.

The function __`KPSS`__ has the `trend` parameter with the following 2 options:
* `c` - **C**onstant trend (default) 
* `ct` - **C**onstant and linear **T**ime trend

Also, the `lag` parameter can be manually added with maximum value being less than the length of the sample data, or automatically set to `12*(nobs/100)**(1/4)`, where `nobs` is the length of the sample size.

__HOW TO INTERPRET__ : 
* If the __p-value__ obtained from the test is __less than the significance level of 0.05__, then we fail to accept the $H_0$ i.e., __the series is non-stationary__.
* If the test statistic is __greater__ than the critical value (for levels of 10%, 5% and 1%), then we accept the $H_0$ i.e., __the series is stationary__, and needs to be differenced.
* If the test statistic __reduces post differencing__, the data should be differenced.

Following code chunk performs kpss test on dataset

df_kpss = pd.DataFrame()
if data_type.lower() == 'panel':
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the result</h2></div>'))
    for each_id in tqdm(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_id)

        # performing KPSS test
        kpss_test = KPSS(panel_data[panel_data[panel_col] == each_id][target_var])

        # assigning whether the null hypothesis is accepted
        stationary_check = 'NO' if kpss_test.pvalue < 0.05 else 'YES'

        # storing the result in dataframe
        df_kpss_output = pd.DataFrame([round(kpss_test.stat,4), round(kpss_test.pvalue,4), 
                                       kpss_test.lags, stationary_check],
                                    index=['Test Statistic','p-value','Number of Lags Used',
                                          'Non-Stationary']).reset_index()
        df_kpss_output.columns = (['Panel','value'])
        df_kpss_output['index'] = each_id

        for k,v in kpss_test.critical_values.items():
            df_kpss_output.loc[df_kpss_output.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),each_id])

        df_kpss_output = df_kpss_output.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
        df_kpss_output.index = [x for x in df_kpss_output.index]

        df_kpss_output.insert(loc=0, column='Panels', value=df_kpss_output.index)
        df_kpss = df_kpss.append(df_kpss_output)
else:
    kpss_test = KPSS(data[target_var])
    # assigning whether the null hypothesis is accepted
    stationary_check = 'NO' if kpss_test.pvalue < 0.05 else 'YES'

    # storing the result in dataframe
    df_kpss_output = pd.DataFrame([round(kpss_test.stat,4), round(kpss_test.pvalue,4), 
                                   kpss_test.lags, stationary_check],
                                index=['Test Statistic','p-value','Number of Lags Used',
                                      'Non-Stationary']).reset_index()
    df_kpss_output.columns = (['Panel','value'])
    df_kpss_output['index'] = "NA"

    for k,v in kpss_test.critical_values.items():
        df_kpss_output.loc[df_kpss_output.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),'NA'])

    df_kpss_output = df_kpss_output.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
    df_kpss_output.index = [x for x in df_kpss_output.index]

    df_kpss_output.insert(loc=0, column='Panels', value=df_kpss_output.index)
    df_kpss = df_kpss.append(df_kpss_output)
clear_output()
# Showing kpss test summary
display(Markdown('__Results of KPSS Test for `{}`:__'.format(target_var)))
if df_kpss.shape[0] == 1:
    display(df_kpss)
else:
    print_table(df_kpss.round(4))


### Semi-parametric test(s)

##### Variance Ratio test

__Variance Ratio test__ is one of the most popular semi-parametric tests to check the _random walk_ hypothesis. Note that the variance ratio test is __NOT a unit root test__. This test is used to check whether the observed series is a _random walk_ or it has some predictability.

* __Null Hypothesis ($H_0$)__ : The series is a ranodm walk series.
* __Alternative Hypothesis ($H_1$)__ : The series is __NOT a random walk__ series.

Rejection of the null with a positive test statistic indicates the presence of positive serial correlation in the time series.

In `trend` parameter of the function __`VarianceRatio`__, `c` allows for a non-zero drift in the random walk, while `nc` requires that the increments to y are of mean 0.

The `lags` must be at least 2, with maximum value that can be added is less than the length of the sample data.

__HOW TO INTERPRET__ : If the __p-value__ obtained from the test is __less than the significance level of 0.05__, then we fail to accept the $H_0$, i.e., __there is no random walk__.

Following code chunk performs variance ratio test on data

if data_type.lower() == 'panel':
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the result</h2></div>'))

    df_var_ratio = pd.DataFrame()
    for each_id in tqdm(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_id)

        # performing the variance ratio test
        var_ratio_test = VarianceRatio(panel_data[panel_data[panel_col] == each_id][target_var])

        # assigning whether the null hypothesis is accepted
        random_walk_check = 'NO' if var_ratio_test.pvalue < 0.05 else 'YES'
        # storing the result in the dataframe
        df_var_ratio_output = pd.DataFrame([round(var_ratio_test.stat,4), round(var_ratio_test.pvalue,4), 
                                            var_ratio_test.lags, random_walk_check],
                                           index=['Test Statistic','p-value','Number of Lags Used','Random Walk']).reset_index()
        df_var_ratio_output.columns = (['Panel','value'])
        df_var_ratio_output['index'] = each_id
        for k,v in var_ratio_test.critical_values.items():
            df_var_ratio_output.loc[df_var_ratio_output.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),each_id])
        df_var_ratio_output = df_var_ratio_output.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
        df_var_ratio_output.index = [x for x in df_var_ratio_output.index]
        df_var_ratio_output.insert(loc=0, column='Panels', value=df_var_ratio_output.index)
        df_var_ratio = df_var_ratio.append(df_var_ratio_output)

else:
    var_ratio_test = VarianceRatio(data[target_var])
    # assigning whether the null hypothesis is accepted
    random_walk_check = 'NO' if var_ratio_test.pvalue < 0.05 else 'YES'
    # storing the result in the dataframe
    df_var_ratio = pd.DataFrame([round(var_ratio_test.stat,4), round(var_ratio_test.pvalue,4), 
                                        var_ratio_test.lags, random_walk_check],index=['Test Statistic','p-value','Number of Lags Used','Random Walk']).reset_index()
    df_var_ratio.columns = (['Panel','value'])
    df_var_ratio['index'] = "NA"
    for k,v in var_ratio_test.critical_values.items():
        df_var_ratio.loc[df_var_ratio.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),"NA"])
    df_var_ratio = df_var_ratio.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
    df_var_ratio.index = [x for x in df_var_ratio.index]
    df_var_ratio.insert(loc=0, column='Panels', value=df_var_ratio.index)
clear_output()
display(Markdown('__Results of Variance Ratio Test for `{}`:__'.format(target_var)))
if df_var_ratio.shape[0] == 1:
    display(df_var_ratio)
else:
    print_table(df_var_ratio.round(4)) 

### Non-Parametric test(s)

##### Phillips-Perron Test

Compared with the ADF test, __Phillips-Perron__ unit root test makes correction to the test statistics and is robust to the unspecified autocorrelation and heteroscedasticity in the errors. There are two types of test statistics, $Z_{\rho}$ and $Z_{\tau}$, which have the same asymptotic distributions as ADF statistic. $Z_{\tau}$ (default) is based on the t-stat and $Z_{\rho}$ uses a test based on the length of time-series (`nobs`) times the re-centered regression coefficient.

* __Null Hypothesis ($H_0$)__ : The series contains __unit root__, and hence it is non-stationary.
* __Alternative Hypothesis ($H_1$)__ : The series is weakly stationary.

The function __`PhillipsPerron`__ has the `trend` parameter with the following 3 options: 
* `nc` - **N**o **C**onstant trend
* `c` - **C**onstant trend (default) 
* `ct` - **C**onstant and linear **T**ime trend

Also, the `lag` parameter can be manually added with maximum value being less than the length of the sample data, or automatically set to `12*(nobs/100)**(1/4)`.

__HOW TO INTERPRET__ : If the __p-value__ obtained from the test is __less than the significance level of 0.05__, then we fail to accept the $H_0$, i.e., __the series is stationary__.

Following code chunk performs the Phillips-Perron test on the dataset 

if data_type.lower() == 'panel':
    # H453700C08, H346600C03, H136400C09, H635900C03
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the result</h2></div>'))

    df_pp = pd.DataFrame()
    for each_id in tqdm(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_id)

        # performing the PP test
        pp_test = PhillipsPerron(panel_data[panel_data[panel_col] == each_id][target_var])
        # assigning whether the null hypothesis is accepted
        stationary_check = 'YES' if pp_test.pvalue < 0.05 else 'NO'
        # storing the result in dataframe
        df_pp_output = pd.DataFrame([round(pp_test.stat,4), round(pp_test.pvalue,4), 
                                     pp_test.lags, stationary_check],
                                    index=['Test Statistic','p-value','Number of Lags Used',
                                          'Stationary']).reset_index()
        df_pp_output.columns = (['Panel','value'])
        df_pp_output['index'] = each_id
        for k,v in pp_test.critical_values.items():
            df_pp_output.loc[df_pp_output.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),each_id])
        df_pp_output = df_pp_output.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
        df_pp_output.index = [x for x in df_pp_output.index]
        df_pp_output.insert(loc=0, column='Panels', value=df_pp_output.index)
        df_pp = df_pp.append(df_pp_output)

else:
    pp_test = PhillipsPerron(data[target_var])
    df_pp = pd.DataFrame()
    # assigning whether the null hypothesis is accepted
    stationary_check = 'YES' if pp_test.pvalue < 0.05 else 'NO'
    # storing the result in dataframe
    df_pp_output = pd.DataFrame([round(pp_test.stat,4), round(pp_test.pvalue,4),pp_test.lags, stationary_check],
                                index=['Test Statistic','p-value','Number of Lags Used','Stationary']).reset_index()
    df_pp_output.columns = (['Panel','value'])
    df_pp_output['index'] = "NA"
    for k,v in pp_test.critical_values.items():
        df_pp_output.loc[df_pp_output.index.max()+1] = (['Critical Value at {}'.format(k),round(v,4),"NA"])
    df_pp_output = df_pp_output.pivot(columns = 'Panel',values = 'value',index = 'index').round(4)
    df_pp_output.index = [x for x in df_pp_output.index]
    df_pp_output.insert(loc=0, column='Panels', value=df_pp_output.index)
    df_pp = df_pp.append(df_pp_output)
clear_output()
display(Markdown('__Results of Phillips-Perron Test for `{}`:__'.format(target_var)))
if df_pp.shape[0] == 1:
    display(df_pp)    
else:
    print_table(df_pp.round(4))

## ACF and PACF

__Auto-Correlation__ and __Partial Auto-Correlation__ are measures of association between current and past series values to indicate which past series values are most useful in predicting future values. These plots can be used to determine the auto-regressive and moving average components of your forecasting model.

* __Auto-Correlation Function (ACF):__ Autocorrelation is the correlation between a signal's observations as a function of the time-lag between them. So, __ACF plot__ is a plot of total correlation between different lag functions. The analysis of autocorrelation is a mathematical tool for finding repeating patterns, such as the presence of a periodic signal obscured by noise.

* __Partial Auto-Correlation Function (PACF):__ PACF plot is a plot of the partial correlation of a stationary time series with its own lagged values, controlling for the values of the time series at all shorter lags.

**NOTE:** Here the plots are generated without differencing the original data.

**How to interpret ACF and PACF:**

| Pattern | Indication in ACF | Indication in PACF |
|:-------:|:---:|:----:|
| Large spike at lag 1 that decreases after a few lags | __AR__ term in the data. Use the PACF to determine the order of the AR term. | __MA__ term in the data. Use the ACF to determine the order of the moving average term. |
| Large spike at lag 1 followed by a decreasing wave that alternates between positive and negative correlations | __Higher order AR__ term in the data. Use the PACF to determine the order of the AR term. | __Higher order MA__ term in the data. Use the ACF to determine the order of the MA term. |
| Significant correlations at the first or second lag, followed by correlations that are not significant | __MA__ term in the data. The number of significant correlations indicates the order of the MA term. | __AR__ term in the data. The number of significant correlations indicates the order of the AR term. |

Following code chunk displays the acf & pacf plots for the data

fig = make_subplots(rows = 2, cols = 1, subplot_titles=('ACF','PACF'))
if data_type.lower() == 'panel':

    all_panel_length = []
    for idx,each_id in enumerate(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_id)
        all_panel_length.append(panel_data.shape[0])

    display(Markdown('__Processing...__'))
    for i, each_id in enumerate(tqdm(selected_panels)):
        vis = True if each_id == selected_panels[0] else False
        panel_data = data.groupby(panel_col).get_group(each_id)
        acf_pacf_plot(panel_data, vis)

    tab_dict_list = []
    display(Markdown('__Generating the tabs...__'))
    for each_id in tqdm(selected_panels):
        vis_check = [[True]*6 if i==each_id else [False]*6 for i in selected_panels]
        vis_check_flat = [i for sublist in vis_check for i in sublist]
        tab_dict_list.append(dict(args = [{"visible": vis_check_flat},
                                            {"title": "ACF-PACF plots for panel: {}".format(each_id)}],
                                    label=each_id, method="update"))

        fig.update_layout(updatemenus=[dict(buttons=list(tab_dict_list),
                                        direction="right", x=0, xanchor="left", y=1.11, yanchor="top")],
                     showlegend=False, title_x=0.5)
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the plots</h2></div>'))
    acf_pacf_plot(data, True)

fig.update_xaxes(title_text='Lags')
clear_output()
display(Markdown('__ACF-PACF plot for column `{}` generated!__'.format(target_var)))
fig.show(config={'displaylogo': False})

## Spectral analysis

Spectral analysis is the decomposition of a time series into underlying __sine and cosine__ functions of different frequencies using __Fourier transform__, which allows us to determine __those frequencies that appear particularly strong or important__. This enables us to __find underlying periodicities__.

The spectral intensities are plotted against time.  

$$\begin{aligned}
x_{t} = \sum_{k} a_{k} sin(2 \pi n ft) + b_{k} cos(2 \pi nft)
\end{aligned}$$

where:  
* `f` is the frequency
* `k` = `1/f` is the period of seasonality
* ($a_{k}$) and ($b_{k}$) are coefficients which can be calculated as $a_{k} = \sum_{t} x_{t} sin(2 \pi n ft)$  and  $b_{k} = \sum_{t} x_{t} cos(2 \pi n ft)$.

The coefficient are usually used to generate spectrum ($s_{k}$) of the data to find out __importance of each frequency (`f`)__. Spectrum is calculated as:

$$\begin{aligned}
s_{k} = \frac{1}{2}(a_{k}^2 + b_{k}^2)
\end{aligned}$$ 

For large ($s_{k}$),  $\frac{k}{n}\$ is important.

__Power-Spectral-Density (PSD)__ analysis is a type of frequency-domain analysis in which a structure is subjected to a probabilistic spectrum of harmonic loading to obtain probabilistic distributions for dynamic response measures.

__Interpretation of the spectral plot__ : The spectral intensity will be plotted against the frequency `f`. The highest spectral intensity is observed, and the corresponding frequency is used to determine the periodicity. The spikes in the spectral plot appear farther along the X axis if the number of seasonal repetitions is greater in the given time period.

fig = go.Figure()
if data_type.lower() == 'panel':

    optimal_freq_pair = {}
    max_psd_pair = {}
    display(Markdown('__Processing...__'))
    for idx,each_id in enumerate(tqdm(selected_panels)):
        panel_data = data.groupby(panel_col).get_group(each_id)
        freqs, psd = periodogram(panel_data[target_var], scaling = 'density',
                                 window=('tukey', 0.25), detrend='linear')
        vis = True if each_id == selected_panels[0] else False

        fig.add_trace(go.Bar(x=freqs, y=psd, width=0.002, visible=vis))
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode='markers',opacity=0.5, visible=vis))
        optimal_freq_pair[each_id] = [f for f,p in zip(freqs,psd) if p == max(psd)][0]
        max_psd_pair[each_id] = max(psd)

    tab_dict_list = []
    display(Markdown('__Generating the tabs...__'))
    for each_id in tqdm(selected_panels):
        vis_check = [[True]*2 if i==each_id else [False]*2 for i in selected_panels]
        vis_check_flat = [i for sublist in vis_check for i in sublist]
        tab_dict_list.append(dict(args = [{"visible": vis_check_flat},
                                         {"title": "PSD plots for <b>panel: {}</b><br>The largest spike of <b>{} PSD</b> was obtained, giving a <b>cycle period of {} {}</b>".format(each_id,round(max_psd_pair[each_id],4), 
                                                                                                                                                                                   round((1/optimal_freq_pair[each_id]),4), 
                                                                                                                                                                                   date_time_freq)}],label=each_id, method="update"))
        fig.update_layout(updatemenus=[dict(buttons=list(tab_dict_list),
                                        direction="right",x=0, xanchor="left", y=1.25, yanchor="top")],showlegend=False,title_x=0.5)
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the plots</h2></div>'))
    freqs, psd = periodogram(data[target_var], scaling = 'density',window=('tukey', 0.25), detrend='linear')

    optimal_freq = [f for f,p in zip(freqs,psd) if p == max(psd)][0]
    fig.add_trace(go.Bar(x=freqs, y=psd, width=0.002))
    fig.add_trace(go.Scatter(x=freqs, y=psd, mode='markers',opacity=0.5, visible=True))
    fig.update_layout(showlegend=False, 
                      title='PSD Plot for <b>{}</b><br>The largest spike of <b>{} PSD</b> was obtained, giving a <b>cycle period of {} {}</b>'.format(target_var, round(max(psd),4), round((1/optimal_freq),4), date_time_freq), title_x=0.5)
fig.update_xaxes(title_text='Frequency')
fig.update_yaxes(title_text= 'PSD')
clear_output()
display(Markdown('__PSD plots generated!__'))
fig.show(config={'displaylogo': False})

# Multivariate EDA

########################################################################################
################################## User Input Needed ###################################
########################################################################################
reg_var = "avgdailyrate" # options for panel -'govt'

if (reg_var == target_var):
   print(colored("\nRegressor and Target variable CANNOT BE SAME!","red",attrs=['bold']))
else:
   clear_output()
   display(Markdown("The column which will be considered as the __regressor__ is: __{}__".format(reg_var)))

## Granger's Causality test

Granger causality is a way to investigate causality between two variables in a time series, i.e., does one variable directly cause the other.

It is based on the idea that if `X` causes `Y`, then the forecast of `Y` based on previous values of `Y` AND the previous values of `X` should outperform the forecast of `Y` based on previous values of `Y` _alone_.

* __Null Hypothesis ($H_0$)__ : The __lagged value__ of a regressor does NOT affect the value of the target variable. So, no causation.
* __Alternate Hypothesis ($H_1$)__ : The lagged value of a regressor affects the value of the target variable.

The __`grangercausalitytests`__ function generated 4 test results:
* `params_ftest`, `ssr_ftest` are based on __F distribution__
* `ssr_chi2test`, `lrtest` are based on __chi-square distribution__

__HOW TO INTERPRET__ : If the __p-value__ obtained from the test is __less than the significance level of 0.05__, then we fail to accept the $H_0$, i.e., __there is causation__.

Following code chunk performs the granger's causality test on the data

df_caus = pd.DataFrame()
if data_type.lower() == 'panel':
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the result</h2></div>'))

    # to find panels with zero values
    zero_panel_ids = data.loc[(data[reg_var] == 0.0) , panel_col]

    for each_id in tqdm(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_id)
        if each_id not in list(zero_panel_ids):
                caus_test = grangercausalitytests(panel_data[[target_var,reg_var]], maxlag= 6)
                caus_test_pvals = [round(caus_test[i+1][0]['params_ftest'][1],4) for i in range(6)]
                caus_test_stats = [round(caus_test[i+1][0]['params_ftest'][0],4) for i in range(6)]
                caus_test_result = list(zip(caus_test_pvals,caus_test_stats))
                caus_test_stat_min = [v for (k,v) in caus_test_result if k == max(caus_test_pvals)][0]
                # assigning whether the null hypothesis is accepted
                caus_check = 'YES' if max(caus_test_pvals) < 0.05 else 'NO'
                # storing the result in a dataframe
                df_caus_output = pd.DataFrame([caus_test_stat_min, max(caus_test_pvals), caus_check],
                                               index=['Test Statistic','p-value', 'Causal']).reset_index()
                df_caus_output.columns = (['Panel','value'])
                df_caus_output['index'] = each_id
                df_caus_output = df_caus_output.pivot(columns = 'Panel',values = 'value',index = 'index')
                df_caus_output.index = [x for x in df_caus_output.index]
                df_caus_output.insert(loc=0, column='Panels', value=df_caus_output.index)
        else: # to skip those columns
            continue

        df_caus = df_caus.append(df_caus_output)

else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the result</h2></div>'))
    caus_test = grangercausalitytests(data[[target_var,reg_var]], maxlag= 6)
    caus_test_pvals = [round(caus_test[i+1][0]['params_ftest'][1],4) for i in range(6)]
    caus_test_stats = [round(caus_test[i+1][0]['params_ftest'][0],4) for i in range(6)]
    caus_test_result = list(zip(caus_test_pvals,caus_test_stats))
    caus_test_stat_min = [v for (k,v) in caus_test_result if k == max(caus_test_pvals)][0]
    # assigning whether the null hypothesis is accepted
    caus_check = 'YES' if max(caus_test_pvals) < 0.05 else 'NO'
    # storing the result in a dataframe
    df_caus_output = pd.DataFrame([caus_test_stat_min, max(caus_test_pvals), caus_check],
                                   index=['Test Statistic','p-value', 'Causal']).reset_index()
    df_caus_output.columns = (['Panel','value'])
    df_caus_output['index'] = "NA"
    df_caus_output = df_caus_output.pivot(columns = 'Panel',values = 'value',index = 'index')
    df_caus_output.index = [x for x in df_caus_output.index]
    df_caus_output.insert(loc=0, column='Panels', value=df_caus_output.index)
    df_caus = df_caus = df_caus.append(df_caus_output)

clear_output()
display(Markdown('__Result of Granger Causality Test between `{}` and `{}`__'.format(target_var, reg_var)))
if df_caus.shape[0] == 1:
    display(df_caus)
else:
    print_table(df_caus.round(4))

## Cointegration test

Let’s define the __order of integration `d`__ which is the number of differencing required to make a non-stationary time series stationary. Consider a pair of time series, both of which are non-stationary. If we take a particular linear combination of theses series, it can sometimes lead to a stationary series. Such a pair of series would then be termed __cointegrated__, and the `d` is less than that of the individual series.

* __Null Hypothesis ($H_0$)__ : There is __NO cointegration__ between the pair of series.
* __Alternative Hypoethesis ($H_1$)__: The pair of series are cointegrated.

The `trend` parameter included in regression for cointegrating equation has 4 options:
* `nc` - `N`o `C`onstant trend
* `c` - `C`onstant trend (default) 
* `ct` - `C`onstant and linear `T`ime trend
* `ctt` - `C`onstant and linear `T`ime and quadratic `T`ime trends

The autolag/lag selection has 3 criterion: `AIC`, `BIC`,`t-stat`

__HOW TO INTERPRET__ : If the __p-value__ obtained from the test is __less than the significance level of 0.05__, then we fail to accept the $H_0$, i.e., __the pair of series are cointegrated__.

df_coint = pd.DataFrame()
if data_type.lower() == 'panel':
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Generating the result</h2></div>'))

    # performing the operation on panel level
    for each_id in tqdm(selected_panels):
        panel_data = data.groupby(panel_col).get_group(each_id)
        lags_given = (panel_data.shape[0]//2) - 1
        # perfroming the cointegration test
        coint_test = coint(panel_data[panel_data[panel_col] == each_id][target_var],
                           panel_data[panel_data[panel_col] == each_id][reg_var], 
                           maxlag=lags_given, autolag='aic')

        # assigning whether the null hypothesis is accepted
        coint_check = 'YES' if coint_test[1] < 0.05 else 'NO'
        coint_test_stats = [round(i,4) for i in list(coint_test[:2])]
        coint_test_stats.append(coint_check)
        coint_conf_intrvl = [round(i,4) for i in coint_test[2].tolist()]
        # storing the result in a dataframe
        df_coint_output = pd.DataFrame(coint_test_stats+coint_conf_intrvl,
                                       index=['Test Statistic','p-value','Cointegrated',
                                              'Critical Value at 1%','Critical Value at 5%', 'Critical Value at 10%']).reset_index()
        df_coint_output.columns = (['Panel','value'])
        df_coint_output['index'] = each_id
        df_coint_output = df_coint_output.pivot(columns = 'Panel',values = 'value',index = 'index')
        df_coint_output.index = [x for x in df_coint_output.index]
        df_coint_output.insert(loc=0, column='Panels', value=df_coint_output.index)
        df_coint = df_coint.append(df_coint_output)
else:
    # perfroming the cointegration test
    coint_test = coint(data[target_var], data[reg_var], maxlag=inp_lag, autolag='aic')
    # assigning whether the null hypothesis is accepted
    coint_check = 'YES' if coint_test[1] < 0.05 else 'NO'
    coint_test_stats = [round(i,4) for i in list(coint_test[:2])]
    coint_test_stats.append(coint_check)
    coint_conf_intrvl = [round(i,4) for i in coint_test[2].tolist()]
    # storing the result in a dataframe
    df_coint_output = pd.DataFrame(coint_test_stats+coint_conf_intrvl,
                                   index=['Test Statistic','p-value','Cointegrated',
                                          'Critical Value at 1%','Critical Value at 5%', 'Critical Value at 10%']).reset_index()
    df_coint_output.columns = (['Panel','value'])
    df_coint_output['index'] = "NA"
    df_coint_output = df_coint_output.pivot(columns = 'Panel',values = 'value',index = 'index')
    df_coint_output.index = [x for x in df_coint_output.index]
    df_coint_output.insert(loc=0, column='Panels', value=df_coint_output.index)
    df_coint = df_coint.append(df_coint_output)
clear_output()
display(Markdown('__Result of Augmented Engle-Granger 2-step Cointegration Test between `{}` and `{}`__'.format(target_var, reg_var)))
if df_coint.shape[0] == 1:
    display(df_coint)
else:
    print_table(df_coint.round(4))

__Forecasting Models__

There are a variety of statistical models or techniques that can be applied to forecasting future datapoints. Each model has it’s own unique interpretation of a time-series. The models below are run on the same sample panel_data. Keep in mind that certain models are more suited to a particular data and will yield better results if tuned appropriately.
There are a variety of metrics that can be used to evaluate a model’s performance, based on which the ideal model can be selected.
Forecast “error” is the difference between an observed value and its forecast. However accuracy measures that are based only on the error term are scale-dependent and cannot be used to make comparisons between series that involve different units. 

The scale-dependent measures are:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

Percentage errors are unit free and more commonly used to measure accuracy. Some common measures are:

* **Mean Absolute Percentage Error (MAPE)**
* **symmetric Mean Absolute Percentage Error (sMAPE)**

The code below is a function that is used to calculate the evaluation metrics mentioned above for each model across all each panels

# User Defined Functions for Modelling

Following code chunk(s) contain required functions for creating time-series model(s) on the dataset

#UDFs

def smape(y_true, y_pred):
    """
    Computes the sMAPE value given the actuals and forecasted numbers
        
    Args:
        y_true (list): Actual numbers
        y_pred (list): Predicted numbers
    
    Returns:
        float: sMAPE value
        
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes MAPE values given the actuals and forecasted numbers
    
    Args:
        y_true (list): Actual numbers
        y_pred (list): Predicted numbers
    
    Returns:
        float: MAPE value
        float: Minimum Absolute Error Percentage
        float: Maximum Absolute Error Percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ae = np.abs((y_true - y_pred) / y_true)
    where_are_inf = np.isinf(ae)
    ae[where_are_inf] = 0
    return np.mean(ae) * 100 , min(ae)*100, max(ae)*100

def get_error_metrics(y_true,y_pred,id_col,m_string,d_string):
    """
    Computes and stores errors metrics in a dictionary
    Args:
        y_true (list): Actual numbers
        y_pred (list): Predicted numbers
        id_col (list): List containing panels names
        m_string (str): Model name
        d_string (str): Dataset type - Validation or Training
    Returns:
        pd.DataFrame: Error metrics
        pd.DataFrame: MAPE metrics
    """
    mape,min_val,max_val = mean_absolute_percentage_error(y_true,y_pred)
    if id_col!='NA':
        error_table_var = pd.DataFrame({'panel_ID': id_col, 'MAPE':mape,
                                        'MSE': mean_squared_error(y_true, y_pred),
                                        'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
                                        'SMAPE': smape(y_true, y_pred),
                                        'model': m_string, 'datatype': d_string},index=[0])
        
        mape_table = pd.DataFrame({'panel_ID': id_col, 'min': min_val, 'MAPE':mape, 'max': max_val,
                                   'model': m_string, 'datatype': d_string}, index=[0])
    else:
        error_table_var = pd.DataFrame({'MAPE':mape,
                                        'MSE': mean_squared_error(y_true, y_pred),
                                        'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
                                        'SMAPE': smape(y_true, y_pred), 
                                        'model': m_string, 'datatype': d_string},index=[0])
        
        mape_table = pd.DataFrame({'min': min_val, 'MAPE':mape, 'max': max_val,
                                   'model': m_string, 'datatype': d_string},index=[0])
    
    return error_table_var,mape_table


def run_model(model_nam,train_df,valid_df,m_string,col='NA'):
    global error_table,valid_mape_table,fitted_model,residuals_train,forecasted_model,residuals_validation
    
    fitted_model = model_nam.predict(train_df[external_variables])
    residuals_train = train_df[target_var] - fitted_model
    forecasted_model = model_nam.predict(valid_df[external_variables])
    residuals_validation = valid_df[target_var] - forecasted_model

    error_local,error_local_mape = get_error_metrics(valid_df[target_var],forecasted_model,
                                                     col,m_string,'Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train_df[target_var],fitted_model,
                                                     col,m_string,'Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
    
    return error_local


def run_model_uv(model_nam,train_df,valid_df,m_string,fit_values,col='NA'):
    global error_table,valid_mape_table,forecasted_model
    
    n_periods_ahead = len(valid_df)
    forecasted_model = model_nam.forecast(n_periods_ahead)
    error_local,error_local_mape = get_error_metrics(valid_df[target_var],forecasted_model, col,
                                                     m_string,'Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train_df[target_var],fit_values, col,
                                                     m_string,'Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
    valid_mape_table.drop_duplicates(keep='last',inplace=True)
    
    return error_local

def error_metrics_plot(model_name,metric):
    """
    Function to plot Validation MAPE on dot and whisker plot
    
    Args:
        model_name (str): Model name
        metric (str): Error metric
    
    Returns:
        None
    """
    global valid_mape_table, panel_ids
    valid_mape_table = valid_mape_table.reset_index(drop = True)
    df_subset = valid_mape_table[(valid_mape_table['model']==model_name) & (valid_mape_table['datatype']=='Validation')].copy()
    df_subset = df_subset.sort_values(by = [metric])
    X = df_subset[metric].round(3)
    Y = df_subset['panel_ID']
    val = round(X.mean(),3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='measured',
                             error_x=dict(type='data', symmetric=False,
                                          array=df_subset['max'],
                                          arrayminus=df_subset['min'],
                                          color='purple', thickness=1.5,
                                          width=3),
                             marker=dict(color='purple', size=8)))
    fig.add_shape(type="line", x0=val,
                  y0=list(df_subset.loc[df_subset[metric] == min(df_subset[metric]), 'panel_ID'])[0],
                  x1=val,
                  y1=list(df_subset.loc[df_subset[metric] == max(df_subset[metric]), 'panel_ID'])[0],
                  line=dict(color="LightSeaGreen", width=4, dash="dashdot"))
    
    if len(panel_ids) < 10:
        plt_ht = 100*len(panel_ids)
    else:
        plt_ht = 1000
    fig.update_layout(title = '{}: Mean Validation {} of panels = {}'.format(model_name,metric,val), 
                      height=plt_ht)
    fig.show(config={'displaylogo': False})

def model_est(error_table, model_name):
    display(Markdown('__The error metrics table:__'))
    show_df = error_table[error_table['model']==model_name].copy()
    show_df = show_df.reset_index(drop = True)
    print_table(show_df.round(4))
    if data_type.lower()=='panel':
        display(Markdown('__The error metrics plot:__'))
        error_metrics_plot(model_name,'MAPE')


def actual_vs_pred(vis, method_name, train_data, valid_data, 
                   target_var, fc_train, fc_valid, panel = 'NA', i = 1):
    """
    Function to generate Actual vs Predicted plot
    
    Args:
        visible (bool): True if trace is to be displayed, False otherwise
        method_name (str): Name of the forecasting method
        train_data (pd.DataFrame): Training dataset
        valid_data (pd.DataFrame): Validation dataset
        target_var (str): Name of the target variable
        fc_train (list): Forecasted numbers on training set
        fc_valid (list): Forecasted numbers on validation set
        panel (str/int): panel ID
        i (int): Flag  variable
    Returns:
        Ploty graph object
    """
    
    i= 1
    global dict_prophet,dict_mars,dict_svr,n_periods_ahead,forecast_panels,fig,pid,data_type
    sub1 = "Actual: Train"
    sub2 = "Forecasted: Train"
    sub3 = "Actual: Validation"
    sub4 = "Forecasted: Validation"
    sub5 = "Split"
    sub7 = "Validation residuals"
    sub6 = "Train residuals"
    
    show = True
    
    ## plotting the actual VS forecasted graph
    fig.add_trace(go.Scatter(x= train_data[date_time_col], y= train_data[target_var], name= sub1,
                             line_color= 'red', mode = 'lines', opacity= 0.8,
                             legendgroup=sub1, showlegend=show, visible=vis), 
                  row = 1, col = i)
    fig.add_trace(go.Scatter(x= train_data[date_time_col], y= fc_train, name= sub2, 
                             line_color= 'maroon', mode = 'lines', opacity= 0.8,
                             legendgroup=sub2, showlegend=show, visible=vis), 
                  row = 1, col = i)
    fig.add_trace(go.Scatter(x= valid_data[date_time_col], y= valid_data[target_var], name= sub3, 
                             line_color= 'lime', mode = 'lines', opacity= 0.8, 
                             legendgroup=sub3, showlegend=show, visible=vis), 
                  row = 1, col = i)
    fig.add_trace(go.Scatter(x= valid_data[date_time_col], y= fc_valid, name= sub4,
                             line_color= 'green', mode = 'lines', opacity=0.8, 
                             legendgroup=sub4, showlegend=show, visible=vis), 
                  row = 1, col = i)
    # Update xaxis,yaxis properties
    fig.update_xaxes(title_text=date_time_col, row = 1, col = i)
    fig.update_yaxes(title_text=target_var, row = 1, col = i)
    
    ## plotting residual graph
    if method_name == 'Prophet':
        y_train = dict_prophet[panel]['Residuals_train'] if (panel != 'NA') else dict_prophet['Residuals_train']
        y_val =  dict_prophet[panel]['Residuals_validation'] if (panel != 'NA') else dict_prophet['Residuals_validation']
    elif method_name == 'MARS':
        y_train = dict_mars[panel]['Residuals_train'] if (panel != 'NA') else dict_mars['Residuals_train']
        y_val =  dict_mars[panel]['Residuals_validation'] if (panel != 'NA') else dict_mars['Residuals_validation']
    elif method_name =='SVR':
        y_train = dict_svr[panel]['Residuals_train'] if (panel != 'NA') else dict_svr['Residuals_train']
        y_val =  dict_svr[panel]['Residuals_validation'] if (panel != 'NA') else dict_svr['Residuals_validation']
    elif method_name =='PLS':
        y_train = dict_pls[panel]['Residuals_train'] if (panel != 'NA') else dict_pls['Residuals_train']
        y_val =  dict_pls[panel]['Residuals_validation'] if (panel != 'NA') else dict_pls['Residuals_validation']  
    else:
        y_train = train_data[target_var]-fc_train
        y_val = valid_data[target_var]-fc_valid
    
    fig.add_trace(go.Scatter(x = train_data[date_time_col], y = y_train, name= sub6, 
                             line_color= 'mediumslateblue', mode = 'lines', opacity= 0.8, 
                             legendgroup=sub6, showlegend=show, visible=vis), 
                  row = 2, col = i)

    fig.add_trace(go.Scatter(x = valid_data[date_time_col], y = y_val, name= sub7, 
                             line_color= 'darkblue', mode = 'lines', opacity= 0.8,
                             legendgroup=sub7, showlegend=show, visible=vis), 
                  row = 2, col = i)
    
    # setting the limit of the plot
    min_ts = min(train_data[date_time_col])
    max_ts = max(valid_data[date_time_col])
    fig.add_shape(type="line", x0=min_ts, y0=0,
                  x1=max_ts, y1=0, line=dict(color="black", width=1, dash="dash"),
                  row = 2, col = i)
    # Update xaxis,yaxis properties
    fig.update_xaxes(title_text=date_time_col, row = 2, col = i)
    fig.update_yaxes(title_text='Residual', row = 2, col = i)

    ## calculating ACF and PACF
    acf_res, acf_conf = acf(y_train , nlags= inp_lag, alpha=.05)
    pacf_res, pacf_conf = pacf(y_train , nlags= inp_lag, alpha=.05)
    # ACF
    fig.add_trace(go.Bar(x= list(range(inp_lag)), y= acf_res.tolist(),
                         marker_color = 'maroon', width = 0.07,
                         showlegend= False,visible=vis), 
                  row = 3, col = i)
    fig.add_trace(go.Scatter(x=list(range(inp_lag)), y=acf_conf[:, 0] - acf_res,
                             line=dict(shape = 'spline', width = 0.01, color='lightgray'),
                             showlegend= False,visible=vis), 
                  row = 3, col = i)
    fig.add_trace(go.Scatter(x=list(range(inp_lag)), y=acf_conf[:, 1] - acf_res,
                             line=dict(shape = 'spline', width = 0.01, color='lightgray'),
                             showlegend= False,visible=vis, fill='tonexty'),
                  row = 3, col = i)
    # Update xaxis,yaxis properties
    fig.update_xaxes(title_text='Lag', row = 3, col = i)
    fig.update_yaxes(title_text='ACF', row = 3, col = i)
    # PACF
    fig.add_trace(go.Bar(x= list(range(inp_lag)), y= pacf_res.tolist(),
                         marker_color = 'maroon', width = 0.07,
                         showlegend= False,visible=vis), 
                  row = 4, col = i)
    fig.add_trace(go.Scatter(x=list(range(inp_lag)), y=pacf_conf[:, 0] - pacf_res,
                             line=dict(shape = 'spline',width = 0.01,color='lightgray'),
                             showlegend= False,visible=vis),
                  row =4, col = i)
    fig.add_trace(go.Scatter(x=list(range(inp_lag)), y=pacf_conf[:, 1] - pacf_res,
                             line=dict(shape = 'spline',width = 0.01,color='lightgray'),
                             showlegend= False,visible=vis, fill='tonexty'),
                  row = 4, col = i)
    # Update xaxis,yaxis properties
    fig.update_xaxes(title_text='Lag', row = 4, col = i)
    fig.update_yaxes(title_text='PACF', row = 4, col = i)

    # Use date string to set xaxis range
    if data_type.lower()=='panel':
        fig.update_layout(height=1500, width= 800, legend_orientation="h", xaxis_range = [min_ts, max_ts],
                          title_text = "{} :<br>Row 1: Actual vs Forecasted, Row 2: Residuals vs Time<br>Row 3: Residuals ACF, Row 4: Residual PACF".format(method_name),
                          title_x = 0.5)
    else:
         fig.update_layout(height=1500, width= 800, legend_orientation="h", xaxis_range = [min_ts, max_ts],
                           title_text = "{}".format(method_name), title_x = 0.5)
    return fig

def fit_tabs(selected_panels,model_name = None,forecast_flag = None):
    """
    Generates buttons in the horizontal dropdown for the selected panels for the goodness of fit section
    Args:
        selected_panels (list): List of panels to generate buttons for
        model_name (str): Name of the forecasting model to filter data for
    Returns:
        None
    """
    # performing the operation on panel level
    panel_dict_list = []
    
    # creating the drop-down option
    display(Markdown('__Generating results in tabs...__'))
    for each_panel in tqdm(selected_panels):
        if forecast_flag != True:
            vis_check = [[True]*12 if i==each_panel else [False]*12 for i in selected_panels]
            vis_check_flat = [item for sublist in vis_check for item in sublist]
            mape_train  = valid_mape_table[(valid_mape_table['panel_ID']==each_panel) &
                                           (valid_mape_table['datatype']=="Train") &
                                           (valid_mape_table['model']==model_name)]['MAPE'].round(4).tolist()[0]
            mape_val = valid_mape_table[(valid_mape_table['panel_ID']==each_panel) &
                                        (valid_mape_table['datatype']=="Validation") &
                                        (valid_mape_table['model']==model_name)]['MAPE'].round(4).tolist()[0]
            panel_dict_list.append(dict(args = [{"visible": vis_check_flat},
                                                {"title":"Panel: {}<br>Training MAPE = {}, Validation MAPE = {}<br>Plot 1: Actual vs Forecasted || Plot 2: Residual Plot<br>Plot 3: Residual ACF || Plot 4: Residual PACF".format(each_panel, mape_train, mape_val)}],
                                    label=each_panel, method="update"))
            fig.update_layout(width=950,height=1500)
            fig.update_layout(annotations=[dict(text="Select panel ID", showarrow=False,
                                                x=0, y=1.175, yref="paper", align="left")])
            # Add dropdown by opening the option on horizontal direction
            fig.update_layout(updatemenus=[dict(buttons=list(panel_dict_list),
                                        direction="right",
                                        x=0.1, xanchor="left", y=1.2, yanchor="top")])
        else:
            vis_check = [[True]*5 if i==each_panel else [False]*5 for i in selected_panels]
            vis_check_flat = [item for sublist in vis_check for item in sublist]
            mape_train  = valid_mape_table[(valid_mape_table['panel_ID']==each_panel) & 
                                           (valid_mape_table['datatype']=="Train") &
                                           (valid_mape_table['model']==model_name)]['MAPE'].round(4).tolist()[0]
            panel_dict_list.append(dict(args = [{"visible": vis_check_flat},
                                                {"title": "Panel: {}, Actual vs Forecasted<br>Training MAPE = {}".format(each_panel, mape_train)}],
                                        label=each_panel, method="update"))
            fig.update_layout(updatemenus=[dict(buttons=list(panel_dict_list), direction="right",
                                                x=0, xanchor="left", y=1.25, yanchor="top")])

            
def actual_vs_pred_ex_ante(method_name, train_data, valid_data, target_var, train_pred, valid_pred, 
                           fc_data, fc_pred, panel = None, vis=True):
    """
    Function to generate Actual vs Predicted plot in the Ex Ante section
    
    Args:
        method_name: Name of the forecasting method
        train_data: Train dataset
        valid_data: Validation dataset
        target_var: Name of the target variable
        fc_data: Test data
        fc_pred: Test predicted value
        panel: panel ID
    
    Returns:
        Ploty graph object
    """      
#     global forecast_panels
    
    sub1 = "Actual: Train"
    sub2 = "Predicted: Train"
    sub3 = "Actual: Valid"
    sub4 = "Predicted: Valid"
    sub5 = "Forecasted: Test"

    # plotting the actual train graph
    fig.add_trace(go.Scatter(x= train_data[date_time_col], y= list(train_data[target_var]), name= sub1,
                             line_color= 'turquoise', mode = 'lines', opacity= 0.8,
                             legendgroup=sub1, visible = vis))

    # plotting the predicted train graph
    fig.add_trace(go.Scatter(x= train_data[date_time_col], y= train_pred, name= sub2,
                             mode = 'lines', line_color= 'dodgerblue', opacity= 0.8,
                             legendgroup=sub2, visible = vis))
    
    # plotting the actual valid graph
    fig.add_trace(go.Scatter(x= valid_data[date_time_col], y= list(valid_data[target_var]), name= sub3,
                             line_color= 'blue', mode = 'lines', opacity= 0.8,
                             legendgroup=sub3, visible = vis))

    # plotting the predicted valid graph
    fig.add_trace(go.Scatter(x= valid_data[date_time_col], y= valid_pred, name= sub4,
                             mode = 'lines', line_color= 'darkblue', opacity= 0.8,
                             legendgroup=sub4, visible = vis))

    # plotting the forecasted validation graph
    fig.add_trace(go.Scatter(x= sorted(fc_data[date_time_col]), y= fc_pred, name= sub5, 
                             line_color= 'mediumslateblue', opacity=0.8, mode = 'lines',
                             legendgroup=sub5, visible = vis))

    # setting the limit of the graph
    min_ts = min(sorted(train_data[date_time_col]))
    max_ts = max(sorted(fc_data[date_time_col]))
    
    if panel is None:
        fig_ht = 400
    else:
        fig_ht = 800
    
    fig.update_layout(height=fig_ht, width= 800, legend_orientation="h", 
                      xaxis_range = [min_ts, max_ts], title_x = 0.5,
                      title_text = "{} : Actual vs Forecasted ".format(method_name))
    return fig

# Univariate Time-Series Forecasting <a name = 'univariate'></a>

Univariate forecasting models make predictions based on the historical properties of the dependent variable itself i.e the variable to be forecasted. These models generally incorporate features such as the trend, seasonality and noise of the dependent variable time series.

In order to validate the trained model, we need to forecast n points ahead. The input for that is stored in `n_periods_ahead` variable. By default, we have selected `n_periods_ahead` as same as the length of the validation dataset.

Following code chunks defines variables which are used throughout modelling section

error_table  = pd.DataFrame()
valid_mape_table = pd.DataFrame()
dict_models = {}

if data_type.lower() == 'panel':
    display(Markdown('__Random Panel ID selection__ to generate model summary for panel level data.'))
    pid = random.choice(selected_panels)
    display(Markdown('Panel selected: __{}__'.format(pid)))

## Holt-Winters <a name= 'holtwinters'></a>

Holt Winters is applied to time series that have a strong trend and seasonality component and can be defined using an additive model. It is a triple exponential smoothing technique used to make short term forecasts. The additive equation is as represented below:
                      
$$\begin{aligned}
F_{t+k} = L_{t} + kb_{t} + S_{t+k-s}
\end{aligned}$$

where: 
$\begin{aligned}
L_{t} - Level\;Component   
\end{aligned}$
$\begin{aligned}
b_{t} - Trend\;Component
\end{aligned}$
$\begin{aligned}
S_{t} - Seasonal\;Component 
\end{aligned}$

These three components are computed using alpha[Level], beta[Does exponential smoothing if set to FALSE] and gamma[Seasonal] parameters respectively. These parameters can be adjusted below. 
<br>

Following code chunk takes the time-series data and trains the holt-winters model on it.

## Holt Winters Exponential Smoothing
dict_hw  = {} # Dictionary to store the model objects and results
parameter_table_hw = pd.DataFrame()
track_cell("Holt-Winters model building", flag)
#user ionput
#Specify if addtitive or multiplicative
seasonal_param = 'add'  ## add, mul
#Specify seasonal period
#month - 12, quater- 4, week - 52, day - 365, hour - 60, minute - 3600
seasonal_period_param = 12


clear_output()
if (seasonal_param == 'mul' and sum(n < 0 for n in train[target_var]) != 0):
    display(Markdown("**NOTE:** For multiplicative series data should not be negative."))
else:
    if data_type.lower()=='panel':
        ## Run model for each panel
        for id_col in tqdm(selected_panels):
            valid_grouped = valid.groupby(panel_col).get_group(id_col)
            train_grouped = train.groupby(panel_col).get_group(id_col)
            model_hw = hw.ExponentialSmoothing(train_grouped[target_var], 
                                               seasonal=seasonal_param, 
                                               seasonal_periods=seasonal_period_param).fit()
            run_model_uv(model_hw,train_grouped,valid_grouped,'Holt-Winters',
                         model_hw.fittedvalues,col=id_col)
            dict_hw.update({id_col:{'Model':model_hw, 'Fitted': model_hw.fittedvalues,
                                    'Forecast':forecasted_model}})
            parameter_table_id = pd.DataFrame({'panel_ID': id_col,
                                               'smoothing_level (alpha)': model_hw.params['smoothing_level'].round(3),
                                               'smoothing_slope (beta)': model_hw.params['smoothing_slope'].round(3),
                                               'smoothing_seasonal (gamma)': model_hw.params['smoothing_seasonal'].round(3)},
                                              index=[0])
            parameter_table_hw = parameter_table_hw.append(parameter_table_id)
        clear_output()
        display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">Holt-Winters models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_hw))))
    else:
        display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
        model_hw = hw.ExponentialSmoothing(train[target_var], 
                                           seasonal= seasonal_param, 
                                           seasonal_periods=seasonal_period_param).fit()
        # Forecasting using HW exponential smoothing
        run_model_uv(model_hw,train,valid,'Holt-Winters',model_hw.fittedvalues)
        dict_hw.update({'Model':model_hw, 'Fitted': model_hw.fittedvalues,'Forecast':forecasted_model})
        clear_output()
    dict_models['Holt-Winters'] = dict_hw   
print("Model trained successfully")    

### Model Summary
Following code chunk displays the model coefficients.

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(dict_hw[pid]['Model'].summary().tables[0])
    x0 = dict_hw[pid]['Model'].summary().tables[1].as_html()
    x1 = pd.read_html(x0, header=0, index_col=0)[0]
    x1.insert(loc=0, column='params', value=x1.index)
    print_table(x1.round(4))
else:
    display(dict_hw['Model'].summary())        

### Model parameters for all panels
Following code chunk displays the model summary for each panel

if data_type.lower()=='panel':
    show_df = parameter_table_hw.round(4).reset_index(drop = True)
    print_table(show_df.round(4))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'Holt-Winters')   

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_hw.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name='Holt-Winters',
                            train_data = train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                            valid_data = valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                            fc_train = dict_hw[id_col]['Model'].fittedvalues.reset_index(drop = True),
                            fc_valid = dict_hw[id_col]['Forecast'].reset_index(drop = True),
                            target_var = target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'Holt-Winters')
else:
    actual_vs_pred(vis = True, method_name='Holt-Winters',
                  train_data= train, valid_data = valid, target_var= target_var,
                  fc_train= dict_hw['Model'].fittedvalues.reset_index(drop = True),
                  fc_valid= dict_hw['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

## ARIMA

ARIMA stands for `A`uto`R`egressive `I`ntegrated `M`oving `A`verage model.

* __AR__ part of ARIMA indicates that the evolving variable of interest is regressed on its own lagged (i.e., prior) values.
* __MA__ part indicates that the regression error is actually a linear combination of error terms whose values occurred contemporaneously and at various times in the past.
* __I__ (for “integrated”) indicates that the data values have been replaced with the difference between their values and the previous values (and this differencing process may have been performed more than once). The purpose of each of these features is to make the model fit the data as well as possible.

There are 3 distinct integers `p`, `d`, `q` that are used to parametrize __non-seasonal__ ARIMA models. Because of that, ARIMA models are denoted with the notation `ARIMA(p,d,q)`.

* `p` is the __auto-regressive__ part of the model. It allows us to incorporate the effect of past values into our model. Intuitively, this would be similar to stating that it is likely to be warm tomorrow if it has been warm the past 3 days.
* `d` is the __integrated__ part of the model. This includes terms in the model that incorporate the amount of differencing (i.e. the number of past time points to subtract from the current value) to apply to the time series. Intuitively, this would be similar to stating that it is likely to be same temperature tomorrow if the difference in temperature in the last three days has been very small.
* `q` is the __moving average__ part of the model. This allows us to set the error of our model as a linear combination of the error values observed at previous time points in the past.

__Seasonal__ ARIMA models are usually denoted by `ARIMA(p,d,q)(P,D,Q)m`, where `m` refers to the number of periods in each season.

To know in details about hyperparameters used in this model, refer to this [link](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html).

Following code chunk takes the time-series data and trains the arima model on it.

## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
p = 1                        ## Order of auto-regression (ar) 
d = 0                        ## Order of differencing (integ)
q = 0                        ## Order of moving average (ma)
## ------------------------------------------------------------------------------------------------------------

dict_arima  = {} # Dictionary to store the model objects and results
parameter_table_arima = pd.DataFrame()
global error_table,valid_mape_table,model_arima
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        model_arima = ARIMA(train_grouped[target_var], order= (p,d,q)).fit(disp= -1)

        n_periods_ahead = len(valid_grouped)
        forecasted_arima = model_arima.forecast(steps = n_periods_ahead, 
                                            exog = None, alpha = 0.01)[1]
        error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],forecasted_arima,id_col,'ARIMA','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train_grouped[target_var],model_arima.fittedvalues,id_col,'ARIMA','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)
        dict_arima.update({id_col:{'Model':model_arima,
                                   'Forecast':forecasted_arima,
                                   'Fitted':model_arima.fittedvalues}})

        parameter_table_id = pd.DataFrame({'panel_ID': id_col,
                                           'ar.L1.{}'.format(target_var): model_arima.params['ar.L1.{}'.format(target_var)],
                                           'const': model_arima.params['const']},index=[0])

        parameter_table_arima = parameter_table_arima.append(parameter_table_id)
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">ARIMA models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_arima))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    model_arima = ARIMA(train[target_var], order= (p,d,q)).fit(disp=-1)
    n_periods_ahead = len(valid)
    forecasted_arima = model_arima.forecast(steps = n_periods_ahead, exog = None, alpha = 0.01)[1]
    dict_arima.update({'Model':model_arima, 'Fitted': model_arima.fittedvalues,
                       'Forecast':forecasted_arima})
    error_local,error_local_mape = get_error_metrics(valid[target_var],
                                                     forecasted_arima,'NA','ARIMA','Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train[target_var],
                                                     model_arima.fittedvalues,'NA','ARIMA','Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
    clear_output()
dict_models['ARIMA'] = dict_arima
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(dict_arima[pid]['Model'].summary())
else:
    display(dict_arima['Model'].summary())       ## Display model summary

### Model parameters for all panels
Following code chunk displays the model summary for each panel

if data_type.lower()=='panel':
    show_df = parameter_table_arima.round(4).reset_index(drop = True)
    print_table(show_df.round(4))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')     

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'ARIMA')       

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_arima.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis=visible, method_name='ARIMA',
                             train_data=train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data=valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train=dict_arima[id_col]['Model'].fittedvalues.reset_index(drop = True),
                             fc_valid=pd.Series(dict_arima[id_col]['Forecast']), 
                             target_var=target_var, panel=id_col, i=i)
    clear_output()
    fit_tabs(processed_panels,'ARIMA')
else:
    actual_vs_pred(vis=True,method_name='ARIMA',
                    train_data=train, valid_data=valid.reset_index(drop = True), 
                    target_var=target_var,
                    fc_train= dict_arima['Fitted'].reset_index(drop = True),
                    fc_valid= pd.Series(dict_arima['Forecast']))
clear_output()
fig.show(config={'displaylogo': False})       

## TBATS

TBATS is a univariate, time-series forecasting method that is capable of modeling series with multiple seasonalities. It accommodates trend, seasonality, outliers, auto regressive and moving average components of a time series.

* `T` - Trigonometric Seasonality
* `B` - Box-Cox Transformation
* `A` - ARMA errors [Auto Regressive Moving Average]
* `T` - Trend components
* `S` - Seasonal components

$$\begin{aligned}
y_{t} = l_{t-1} + \phi b_{t-1} + \sum_{i=1}^{T}S_{t-m_{i}}^i + d_{t}
\end{aligned}$$

where:

* `l` is the local level
* `b` is the damped trend
* `d` is the ARMA component for residuals

Following code chunk takes the time-series data and trains the tbats model on it.

## TBATS Method
track_cell("TBATS model building", flag)
dict_tbats  = {} # Dictionary to store the model objects and results
parameter_table_tbats = pd.DataFrame()
## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
use_box_cox = False
use_trend = True
use_damped_trend = True
seasonal_periods = [12]
use_arma_errors = False


tbats_params = TBATS(use_box_cox = use_box_cox, use_trend = use_trend, 
                     use_damped_trend = use_damped_trend, 
                     seasonal_periods = seasonal_periods,
                     use_arma_errors = use_arma_errors)
global error_table,valid_mape_table,model_tbats

if data_type.lower()=='panel':
    ## Run model for each panel    
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)

        model_tbats = tbats_params.fit(train_grouped[target_var])
        n_periods_ahead = len(valid_grouped)
        forecasted_tbats = model_tbats.forecast(steps=n_periods_ahead)

        error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],
                                                         pd.Series(forecasted_tbats),id_col,
                                                         'TBATS','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train_grouped[target_var],
                                                         pd.Series(model_tbats.y_hat),id_col,
                                                         'TBATS','Train')
        error_table = error_table.append(pd.DataFrame(error_local))

        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)

        dict_tbats.update({id_col:{'Model':model_tbats, 'Forecast':pd.Series(forecasted_tbats),
                                   'Fitted': pd.Series(model_tbats.y_hat),
                                   'Residuals':pd.Series(model_tbats.resid)}})
        parameter_table_id = pd.DataFrame({'panel_ID': id_col,
                                           'Smoothing (alpha)':model_tbats.params.alpha,
                                           'Trend (beta)': model_tbats.params.beta,                                  
                                           'Damping (phi)': model_tbats.params.phi},index=[0])
        parameter_table_tbats = parameter_table_tbats.append(parameter_table_id)

    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">TBATS models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_tbats))))
else:
    model_tbats = tbats_params.fit(train[target_var])
    n_periods_ahead = len(valid)
    forecasted_tbats = model_tbats.forecast(steps= n_periods_ahead)
    dict_tbats.update({'Model':model_tbats,'Forecast':pd.Series(forecasted_tbats),
                       'Fitted': pd.Series(model_tbats.y_hat),
                       'Residuals':pd.Series(model_tbats.resid)})

    error_local,error_local_mape = get_error_metrics(valid[target_var],pd.Series(forecasted_tbats),'NA','TBATS','Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train[target_var],pd.Series(model_tbats.y_hat),'NA','TBATS','Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
dict_models['TBATS']=dict_tbats
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    print(dict_tbats[pid]['Model'].summary())
else:
    print(dict_tbats['Model'].summary())       

### Model parameters for all panels
Following code chunk displays the model summary for each panel

if data_type.lower()=='panel':
    show_df = parameter_table_tbats.round(4).reset_index(drop = True)
    print_table(show_df.round(4))
else:
    show_df = pd.DataFrame()
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'TBATS')  

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_tbats.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name= 'TBATS',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data = valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= pd.Series(dict_tbats[id_col]['Model'].y_hat),
                             fc_valid= dict_tbats[id_col]['Forecast'],
                             target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'TBATS')
else:
    actual_vs_pred(vis = True, method_name= 'TBATS',
                   train_data= train, valid_data = valid.reset_index(drop = True), 
                   target_var= target_var,
                   fc_train= pd.Series(dict_tbats['Model'].y_hat), 
                   fc_valid= dict_tbats['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

## ETS

ETS is a univariate forecasting model that uses exponential smoothing to estimate trend and seasonal components of the time series. The advantage of using this model is it’s flexibility while making estimations for different types of traits.

Following code chunk takes the time-series data and trains the ets model on it.

track_cell("ETS model building", flag)
dict_ets = {} # Dictionary to store the model objects and results
parameter_table_ets = pd.DataFrame()
## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
exponential_trend=False
damped_trend=False

global error_table,valid_mape_table,model_ets
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        model_ets = hw.Holt(train_grouped[target_var],
                            exponential=exponential_trend,
                            damped=damped_trend).fit()
        run_model_uv(model_ets,train_grouped,valid_grouped,'ETS',model_ets.fittedvalues,col=id_col)
        dict_ets.update({id_col:{'Model':model_ets,
                                'Fitted': model_ets.fittedvalues,
                                'Forecast':forecasted_model}})                                  
        parameter_table_id = pd.DataFrame({'panel_ID': id_col,
                                           'smoothing_level (alpha)': model_ets.params['smoothing_level'].round(3),
                                           'smoothing_slope (beta)': model_ets.params['smoothing_slope'].round(3),
                                           'smoothing_seasonal (gamma)': model_ets.params['smoothing_seasonal'].round(3)},index=[0])
        parameter_table_ets = parameter_table_ets.append(parameter_table_id)
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">ETS models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_ets))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    model_ets = hw.Holt(train[target_var],
                        exponential=exponential_trend,
                        damped=damped_trend).fit()
    # Forecasting using ETS
    run_model_uv(model_ets,train,valid,'ETS',model_ets.fittedvalues)
    dict_ets.update({'Model':model_ets, 'Fitted': model_ets.fittedvalues, 'Forecast':forecasted_model})
    clear_output()
dict_models['ETS']=dict_ets 
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(dict_ets[pid]['Model'].summary())
else:
    display(dict_ets['Model'].summary())

### Model parameters for all panels
Following code chunk displays the model summary for each panel

if(data_type.lower()=='panel'):
    show_df = parameter_table_ets.round(4).reset_index(drop = True)
    print_table(show_df.round(4))
else:
    show_df = pd.DataFrame()
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')  

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'ETS')      

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_ets.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis= visible,method_name= 'ETS',
                            train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                            valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                            fc_train= dict_ets[id_col]['Model'].fittedvalues.reset_index(drop = True),
                            fc_valid= dict_ets[id_col]['Forecast'].reset_index(drop = True), 
                            target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'ETS')
else:
    actual_vs_pred(vis= True,method_name= 'ETS',
                   train_data= train, valid_data= valid, target_var= target_var,
                   fc_train= dict_ets['Fitted'], 
                   fc_valid= dict_ets['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

# Multivariate Forecasting

## ARIMAX

To know in details about hyperparameters used in this model, refer to this [link](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html).

Following code chunk takes the time-series data and trains the arimax model on it.

## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
p = 1                        ## Order of auto-regression (ar) 
d = 0                        ## Order of differencing (integ)
q = 0                        ## Order of moving average (ma)
## ------------------------------------------------------------------------------------------------------------

## ARIMAX
dict_arimax  = {} # Dictionary to store the model objects and results
parameter_table_arimax = pd.DataFrame()

global error_table,valid_mape_table,model_arimax
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        model_arimax = ARIMA(train[train[panel_col] == id_col][target_var], 
                             exog= train[train[panel_col] == id_col][external_variables],
                             order= (p,d,q)).fit(trend='nc')
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)

        n_periods_ahead = len(valid_grouped)
        forecasted_arimax = model_arimax.forecast(n_periods_ahead, 
                                                  valid_grouped[external_variables],
                                                  alpha = 0.01)[1] ## Extrating forecasted values
        error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],forecasted_arimax,id_col,'ARIMAX','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train_grouped[target_var],model_arimax.fittedvalues,id_col,'ARIMAX','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)
        parameter_table_id = pd.DataFrame({'panel_ID': id_col,
                                           external_variables[0]: model_arimax.params[external_variables[0]],                                          
                                           external_variables[1]: model_arimax.params[external_variables[1]],
                                           external_variables[2]: model_arimax.params[external_variables[2]],
                                           external_variables[3]: model_arimax.params[external_variables[3]],
                                           'ar.L1.{}'.format(target_var): model_arimax.params['ar.L1.{}'.format(target_var)]},index=[0])

        parameter_table_arimax = parameter_table_arimax.append(parameter_table_id)         
        dict_arimax.update({id_col:{'Model':model_arimax,
                                   'Forecast':forecasted_arimax,
                                   'Fitted':model_arimax.fittedvalues}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">ARIMAX models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_arimax))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    model_arimax = ARIMA(train[target_var], exog= train[external_variables], 
                         order= (p,d,q)).fit(trend='nc')
    ## Forecasting using ARIMA
    n_periods_ahead = len(valid)
    forecasted_arimax = model_arimax.forecast(n_periods_ahead, valid[external_variables], alpha = 0.01)[1]
    dict_arimax.update({'Model':model_arimax,
                        'Fitted': model_arimax.fittedvalues,
                        'Forecast':forecasted_arimax})
    error_local,error_local_mape = get_error_metrics(valid[target_var],forecasted_arimax,'NA',
                                                     'ARIMAX','Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train[target_var],model_arimax.fittedvalues,'NA',
                                                     'ARIMAX','Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
    clear_output()
dict_models.update({'ARIMAX':dict_arimax})       
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(dict_arimax[pid]['Model'].summary())
else:
    display(dict_arimax['Model'].summary())       ## Display model summary       

### Model parameters for all panels
Following code chunk displays the model summary for each panel

if(data_type.lower()=='panel'):
    show_df = parameter_table_arimax.round(4).reset_index(drop = True)
    print_table(show_df.round(4))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

### Model Estimation
Model metrics for training and validation data

model_est(error_table, 'ARIMAX')

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_arimax.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis= visible,method_name= 'ARIMAX',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= dict_arimax[id_col]['Model'].fittedvalues.reset_index(drop = True),
                             fc_valid= pd.Series(dict_arimax[id_col]['Forecast']),
                             target_var= target_var, panel = id_col, i= i)     
    clear_output()
    fit_tabs(processed_panels,'ARIMAX')
else:
     actual_vs_pred(vis = True, method_name= 'ARIMAX',
                    train_data= train, valid_data= valid.reset_index(drop = True), 
                    target_var= target_var,
                    fc_train= pd.Series(dict_arimax['Model'].fittedvalues),
                    fc_valid= pd.Series(dict_arimax['Forecast']))
clear_output()
fig.show(config={'displaylogo': False}) 

## Prophet

Prophet is an open source forecasting tool by Facebook which run on an additive regression model with four main components:

* A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
* A yearly seasonal component modeled using Fourier series.
* A weekly seasonal component using dummy variables.
* A user-provided list of important holidays.

Prophet is mainly used to overcome the complexity introduced by variety of forecasting problems and getting reliable forecasts from that. 
The forecasts produced by prophet with default parameters is often considered as reliable as those produced by a skilled data scientist.
Although this framework of forecasting is business use-case oriented, in order to completely take leverage of the capabilities of Prophet, 
the user should make sure the observed data has the following characteristics:

* Hourly, daily, or weekly observations with at least a few months (preferably a year) of history
* Strong multiple “human-scale” seasonalities: day of week and time of year
* Important holidays that occur at irregular intervals that are known in advance
* A reasonable number of missing observations or large outliers
* Historical trend changes, for instance due to product launches or logging changes
* Trends that are non-linear growth curves, where a trend hits a natural limit or saturates

For further information visit [this](https://facebook.github.io/prophet/docs/quick_start.html#python-api) link.

 __Prophet model parameter definitions :__<br>
<br>__k (Mx1 array)__: M posterior samples of the initial slope.
<br>__m (Mx1 array)__: The initial intercept. 
<br>__delta (MxN array)__: The slope change at each of N changepoints. 
<br> __beta (MxK matrix)__: Coefficients for K seasonality features.
<br>__sigma_obs (Mx1 array)__: Noise level.

Following code chunk takes the time-series data and trains the prophet model on it.

## Build the models 

track_cell("Prophet model building", flag)
dict_prophet = {}
parameter_table_prophet = pd.DataFrame()
global error_table,valid_mape_table,model_prophet,forecasted_prophet
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        prophet_data = train.copy()
        prophet_data.rename(columns = {date_time_col:'ds', target_var:'y'}, inplace = True)

        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = prophet_data.groupby(panel_col).get_group(id_col)

        ## ------------------------------------------------------------------------------------------------------------
        # `yearly_seasonality=True` or 
        # `weekly_seasonality=True` or
        # `daily_seasonality=True` depending on the peridocity
        model_prophet = Prophet()

        model_prophet = model_prophet.fit(train_grouped)   ## Fitting on observed data
        n_periods_ahead = len(valid_grouped)
        future_prophet_data = model_prophet.make_future_dataframe(periods = n_periods_ahead,
                                                                  include_history=True,
                                                                  freq=date_time_freq)
        forecasted_prophet = model_prophet.predict(future_prophet_data)  ## Forecasting using prophet

        error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],
                                                         forecasted_prophet.yhat[len(train_grouped):],
                                                         id_col,'Prophet','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train_grouped['y'],
                                                         forecasted_prophet.yhat[:len(train_grouped)],
                                                         id_col,'Prophet','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)
        parameter_table_id = pd.DataFrame({'panel_ID': id_col,
                                           'k':np.asscalar(model_prophet.params['k']),
                                           'm': np.asscalar(model_prophet.params['m']),
                                           'sigma_obs': np.asscalar(model_prophet.params['k'])},
                                           index=[0])
        parameter_table_prophet = parameter_table_prophet.append(parameter_table_id)

        dict_prophet.update({id_col:{'Model':model_prophet,
                                    'Forecast':forecasted_prophet.yhat[len(train_grouped):],
                                    'Forecast_df': forecasted_prophet,
                                    'Residuals_train':model_prophet.history['y'].reset_index(drop = True)-forecasted_prophet.yhat[:len(train_grouped)].reset_index(drop = True),
                                    'Residuals_validation':valid_grouped[target_var].reset_index(drop = True)-forecasted_prophet.yhat[len(train_grouped):].reset_index(drop = True),
                                    'Fitted': forecasted_prophet.yhat[:len(train_grouped)]}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">Prophet models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_prophet))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    prophet_data = train.copy()
    prophet_data.rename(columns = {date_time_col:'ds', target_var:'y'}, inplace = True)
    ## ------------------------------------------------------------------------------------------------------------
    # `yearly_seasonality=True` or 
    # `weekly_seasonality=True` or
    # `daily_seasonality=True` depending on the peridocity
    model_prophet = Prophet(weekly_seasonality= True)
    model_prophet = model_prophet.fit(prophet_data)   ## Fitting on observed data
    n_periods_ahead = len(valid)
    future_prophet_data = model_prophet.make_future_dataframe(periods = n_periods_ahead,
                                                              include_history=True,
                                                              freq=date_time_freq)

    forecasted_prophet = model_prophet.predict(future_prophet_data)  
    dict_prophet.update({'Model':model_prophet,
                        'Forecast':forecasted_prophet.yhat[len(train):],
                        'Forecast_df': forecasted_prophet,
                        'Residuals_train':model_prophet.history['y'].reset_index(drop = True)-forecasted_prophet.yhat[:len(train)].reset_index(drop = True),
                        'Residuals_validation':valid[target_var].reset_index(drop = True)-forecasted_prophet.yhat[len(train):].reset_index(drop = True),
                        'Fitted': forecasted_prophet.yhat[:len(train)]})
    error_local,error_local_mape = get_error_metrics(valid[target_var],forecasted_prophet.yhat[len(train):],'NA','Prophet','Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train[target_var],forecasted_prophet.yhat[:len(train)],'NA','Prophet','Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
    clear_output()
dict_models.update({'Prophet':dict_prophet})
print("Model trained successfully")

if(data_type.lower()=='panel'):
    show_df = parameter_table_prophet.round(3).reset_index(drop = True)
    print_table(show_df.round(4))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')        

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'Prophet')   

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_prophet.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name= 'Prophet',
                             train_data= train.groupby(panel_col).get_group(id_col),
                             valid_data= valid.groupby(panel_col).get_group(id_col),
                             fc_train= dict_prophet[id_col]['Fitted'],
                             fc_valid= dict_prophet[id_col]['Forecast'],
                             target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'Prophet')
else:
    actual_vs_pred(vis = True,method_name= 'Prophet', target_var= target_var,
                   train_data= train, valid_data= valid,
                   fc_train= dict_prophet['Fitted'],
                   fc_valid= dict_prophet['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

## UCM

The UCM procedure analyzes and forecasts equally spaced univariate time series data using the Unobserved Components Model (UCM). A UCM decomposes a response series into components such as trend, seasonal, cycle, and the regression effects due to predictor series. These components capture the salient features of the series that are useful in explaining and predicting its behavior. The UCMs are also called Structural Models in the time series literature.

$$\begin{aligned}
y_{t} = \mu_{t} + \gamma_{t} + \psi_{t} + \sum_{j=1}^{m} \beta_{j}x_{jt} + \epsilon_{t}
\end{aligned}$$

where:  

* μ is the trend component  
* γ is the seasonal component
* ψ is the cyclic component 
* β is the contribution of regressors  

Following code chunk takes the time-series data and trains the ucm model on it.

# Function call to train the models
track_cell("UCM model building", flag)
dict_ucm  = {} # Dictionary to store the model objects and results
parameter_table_ucm = pd.DataFrame()

global error_table,valid_mape_table,model_ucm
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)

        model_ucm = smt.UnobservedComponents(train_grouped[target_var],
                                             exog=train_grouped[external_variables], 
                                             autoregressive=1,
                                             seasonal = enter_freq,
                                             cycle=True, stochastic_cycle=True)
        fit_ucm = model_ucm.fit(method='powell', disp= False)
        forecasted_ucm = fit_ucm.get_forecast(len(valid_grouped[external_variables]),
                                              exog = valid_grouped[external_variables])
        error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],forecasted_ucm.summary_frame()['mean'],id_col,'UCM','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train_grouped[target_var],fit_ucm.fittedvalues,id_col,'UCM','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)
        parameter_table_id = pd.DataFrame({'panel_ID': id_col,
                                           'beta.{}'.format(external_variables[0]): fit_ucm.params['beta.{}'.format(external_variables[0])],
                                           'beta.{}'.format(external_variables[1]): fit_ucm.params['beta.{}'.format(external_variables[1])],
                                           'sigma2.cycle': fit_ucm.params['sigma2.cycle'],
                                           'frequency.cycle': fit_ucm.params['frequency.cycle']},index=[0])

        parameter_table_ucm = parameter_table_ucm.append(parameter_table_id)     
        dict_ucm.update({id_col:{'Model':fit_ucm,
                                'Forecast':forecasted_ucm.summary_frame()['mean'],
                                'Fitted':fit_ucm.fittedvalues}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">UCM models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_ucm))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    model_ucm =  smt.UnobservedComponents(train[target_var],exog=train[external_variables], 
                                          cycle=True, stochastic_cycle=True)
    fit_ucm = model_ucm.fit(method= 'powell', disp= False)
    forecasted_ucm = fit_ucm.get_forecast(len(valid[external_variables]),
                       exog = valid[external_variables])       
    dict_ucm.update({'Model':fit_ucm,
                    'Forecast':forecasted_ucm.summary_frame()['mean'],
                    'Fitted':fit_ucm.fittedvalues})       
    error_local,error_local_mape = get_error_metrics(valid[target_var],forecasted_ucm.summary_frame()['mean'],'NA','UCM','Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train[target_var],fit_ucm.fittedvalues,'NA','UCM','Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
    clear_output()
dict_models.update({'UCM':dict_ucm})
print("Model trained successfully")    

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(dict_ucm[pid]['Model'].summary())
else:
    display(dict_ucm['Model'].summary())       ## Display model summary    

### Model parameters for all panels
Following code chunk displays the model summary for each panel

 if data_type.lower()=='panel':
    show_df = parameter_table_ucm.round(3).reset_index(drop = True)
    print_table(show_df.round(4))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'UCM')

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_ucm.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name='UCM',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= dict_ucm[id_col]['Model'].fittedvalues.reset_index(drop = True),
                             fc_valid= dict_ucm[id_col]['Forecast'].reset_index(drop = True), 
                             target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'UCM')
else:
    fig = actual_vs_pred(method_name='UCM', vis = True,
                         train_data= train, valid_data= valid, target_var= target_var,
                         fc_train=dict_ucm['Model'].fittedvalues,
                         fc_valid=dict_ucm['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

## Ordinary Least Squares Regression

Linear Regression is a technique that attempts to model a linear relationship between the dependent(y) variable and one or more external variables(regressors). This relationship is best described by the following equation:

$$\begin{aligned}
y_{t} = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + ...
\end{aligned}$$

where:  
<br>

* β0 is the intercept coefficient
* β1 and β2 are slope coefficients
* X0, X1 are Regressors  

Following code chunk takes the time-series data and trains the ols model on it.

# Function call to build all the models
track_cell("OLS model building", flag)
dict_ols = {} # Dictionary to store the model objects and results
coefficient_table_ols  = pd.DataFrame()
global error_table,valid_mape_table,model_ols,forecasted_model

if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)

        model_ols = smf.ols(formula, data=train_grouped).fit()
        fit_vals=model_ols.fittedvalues

        forecasted_model = model_ols.predict(valid_grouped)
        error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],forecasted_model,
                                                         id_col,'OLS','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train_grouped[target_var],fit_vals,
                                                         id_col,'OLS','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)


        dict_ols.update({id_col:{'Model':model_ols,
                                'Fitted': model_ols.fittedvalues,
                                'Forecast':forecasted_model}})
        err_series = model_ols.bse
        coef_df = pd.DataFrame({'coef': model_ols.params.values[1:],'err': err_series.values[1:],
                                'varname': err_series.index.values[1:],'panelID': id_col})
        coefficient_table_ols = coefficient_table_ols.append(coef_df)
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">OLS Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_ols))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    model_ols = smf.ols(formula, data=train).fit()
    fit_vals = model_ols.fittedvalues
    col='NA'

    forecasted_model = model_ols.predict(valid)
    error_local,error_local_mape = get_error_metrics(valid[target_var],forecasted_model,
                                                     col,'OLS','Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train[target_var],fit_vals,
                                                     col,'OLS','Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)

    dict_ols.update({'Model':model_ols,
                    'Fitted': model_ols.fittedvalues,
                    'Forecast':forecasted_model})
    clear_output()
dict_models.update({'OLS':dict_ols})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(dict_ols[pid]['Model'].summary())
else:
    display(dict_ols['Model'].summary())       ## Display model summary

### Model fit on the entire dataset available [Naive Model]:

The model is also trained on the entire dataset to serve as a basis for whether a panel level or naive model would be a better fit for each panel

model_ols = smf.ols(formula, data=train).fit()
naive_coef = model_ols.params
display(model_ols.summary())

### Coefficients plot
*  These plots are helpful in estimating the need for panel-wise models

If the coefficents of the panel wise models allign with the dotted line <b>coefficent of naive model</b>, the naive model can be used to model all panels
If any of the panel coefficents is distant from the dotted line <b>coefficent of naive model</b>, that panel requires it’s own model

if data_type.lower()=='panel':
    title_list = []
    variable_list = list(naive_coef.index[1:])
    for i in variable_list:
        x = '{}<br>Naive value = {}'.format(i,round(naive_coef[i],4))
        title_list.append(x)

    title_tuple = tuple(title_list)
    naive_coef = naive_coef.round(3)
    fig = make_subplots(rows=1, cols=len(variable_list), subplot_titles=title_tuple)
    for j in variable_list:
        df_subset = coefficient_table_ols[coefficient_table_ols['varname']==j]
        df_subset = df_subset.reset_index(drop = True)
        i = variable_list.index(j)+1
        fig = coef_plot(df_subset,i=i,val = naive_coef[i].round(3),l=len(variable_list))
    fig.show(config={'displaylogo': False})       
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'OLS')

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

# Actual vs Predicted plot
track_cell("OLS goodness of fit", flag)
if __name__ == '__main__':
    fig = make_subplots(rows=4, cols=1, 
                        subplot_titles=['Actual vs Predicted','Residual vs Time',
                                        'Residual ACF','Residual PACF'],
                        vertical_spacing = 0.1)
    if data_type.lower() == "panel":
        display(Markdown('__Processing...__'))
        processed_panels = dict_ols.keys()
        for i,id_col in enumerate(tqdm(processed_panels)):
            visible = True if id_col == list(processed_panels)[0] else False
            fig = actual_vs_pred(vis = visible,method_name= 'OLS',
                                 train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                                 valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                                 fc_train= dict_ols[id_col]['Model'].fittedvalues.reset_index(drop = True),
                                 fc_valid= dict_ols[id_col]['Forecast'].reset_index(drop = True),
                                 target_var= target_var, panel = id_col, i= i)
        clear_output()
        fit_tabs(processed_panels,'OLS')
    else:
        fig = actual_vs_pred(method_name='OLS', vis=True,
                             train_data= train, valid_data = valid, target_var= target_var,
                             fc_train= dict_ols['Model'].fittedvalues,
                             fc_valid= dict_ols['Forecast'])
    clear_output()
    fig.show(config={'displaylogo': False})
        
else:
    pass

## MARS

Multivariate Adaptive Regression Splines is a non-parametric machine learning machine learning technique which captures non-linearity aspects of a 
polynomial regression by assessing cut-points or knots (very similar to step functions). 
The procedure assesses each data point for each predictor as a knot and creates a linear regression model with the 
candidate feature(s) using a hinge function. In other words, The MARS algorithm builds a model in two steps. First, 
it creates a collection of so-called basis functions (BF). In this procedure, the range of predictor values is partitioned 
in several groups. For each group, a separate linear regression is modeled, each with its own slope. The connections between 
the separate regression lines are called knots. The MARS algorithm automatically searches for the best spots to place the knots.
Each knot has a pair of basis functions. These basis functions describe the relationship between the environmental variable and
the response.


To know in details about hyperparameters used in this model, refer to this [link](https://contrib.scikit-learn.org/py-earth/).
Following code chunk takes the time-series data and trains the mars model on it.

track_cell("MARS model building", flag)
dict_mars  = {} # Dictionary to store the model objects and results
## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
max_degree=3
penalty=1.0 
minspan_alpha = 0.1 
endspan_alpha = 0.01 
endspan=5
## ------------------------------------------------------------------------------------------------------------
model_mars = Earth(max_degree=max_degree,
                   penalty=penalty,
                   minspan_alpha = minspan_alpha,
                   endspan_alpha = endspan_alpha,
                   endspan=5)
global error_table,valid_mape_table
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_mars.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'MARS',col=id_col)
            dict_mars.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,
                                    'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,
                                    'Residuals_validation':residuals_validation}})

    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">MARS models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_mars))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_mars.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'MARS')
    dict_mars.update({'Model':fit_model,
                    'Forecast':forecasted_model,
                    'Fitted': fitted_model,
                    'Residuals_train': residuals_train,
                    'Residuals_validation':residuals_validation})
    clear_output()
dict_models.update({'MARS':dict_mars})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    mars_res = dict_mars[pid]['Model'].summary().split('\n')
else:
    mars_res = dict_mars['Model'].summary().split('\n')

display(Markdown('__{}__'.format(mars_res[0])))
results = mars_res[4:-2]
results_list = []
for res in results:
    res_ = ', '.join(res.split())
    final_re = res_.split(',')
    results_list.append(final_re)

result_df = pd.DataFrame(columns=['Basis Function','Pruned','Coefficient'],
                        data = results_list)

result_df['Coefficient'] = [np.nan if i==' None' else float(i) for i in result_df['Coefficient']]
result_df['Coefficient'] = result_df['Coefficient'].astype(float).round(4)

fig = go.Figure(data=[go.Table(columnwidth = [400,100,100], header=dict(values=list(result_df.columns),
                                                       fill_color='grey',align='center',
                                                       font=dict(color='white')),
                                           cells=dict(values=[result_df[i] for i in result_df.columns],
                                                      fill_color='rgba(0,191,255,0.5)',
                                                      align='center', font=dict(color='black')), 
                                           visible = True)])
fig.update_layout(height=300,
                  margin=dict(l=0,r=0,b=0,t=0,pad=0))
fig.show(config={'displaylogo': False})

### Spline plot

Spline Regression is a non-parametric regression technique. This regression technique divides the datasets into bins at intervals or points called knots and each bin has its separate fit. Below are the Spline plots for all the selected external variables. These plots indicate the nonlinearity present in the dataset while predicting your target variable. The number of knots control the degree of nonlinearity in the dataset.

fig = make_subplots(rows = 1, cols = len(external_variables), 
                    subplot_titles= tuple(external_variables), y_title=target_var)
display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
for i,col in enumerate(external_variables):
    show = True if col==external_variables[0] else False 
    if(data_type.lower()=='panel'):        
        train_data = train[train[panel_col]==pid][external_variables]
        train_mean = train_data.apply(lambda x : train_data.mean(), axis = 1)
        train_mean[col] = np.linspace(min(train_data[col]),max(train_data[col]), len(train_data))
        predictions = dict_mars[pid]['Model'].predict(train_mean)
        y_actual = train[train[panel_col]==pid][target_var]      
    else:
        train_data = train[external_variables]
        train_mean = train_data.apply(lambda x : train_data.mean(), axis = 1)
        train_mean[col] = np.linspace(min(train_data[col]),max(train_data[col]), len(train_data))
        predictions = dict_mars['Model'].predict(train_mean)
        y_actual = train[target_var]
    fig.add_trace(go.Scatter(x= train_mean[col],y= y_actual,
                    name= 'actual',line_color= 'red',mode = 'markers',                   
                    opacity= 0.8,legendgroup='actual',showlegend=show, visible=True),
                    row  = 1, col = i+1)
    fig.add_trace(go.Scatter(x= train_mean[col],y= predictions,
                    name= 'predicted',line_color= 'blue',mode = 'lines',                   
                    opacity= 0.8,legendgroup='predicted',showlegend=show, visible=True),
                    row  = 1, col = i+1)
    fig.update_layout(height=400, width= 400*len(external_variables),legend_orientation = 'h')
clear_output()
fig.show(config={'displaylogo': False})

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'MARS')      

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_mars.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name='MARS',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= pd.Series(dict_mars[id_col]['Fitted']),
                             fc_valid= pd.Series(dict_mars[id_col]['Forecast']),
                             target_var= target_var, panel = id_col,i= i)
    clear_output()
    fit_tabs(processed_panels,'MARS')
else:
    fig = actual_vs_pred(method_name='MARS', vis=True,
                         train_data= train, valid_data= valid, target_var= target_var,
                         fc_train= pd.Series(dict_mars['Fitted']),
                         fc_valid= pd.Series(dict_mars['Forecast']))

clear_output()
fig.show(config={'displaylogo': False})

## VAR

Vector Auto Regression (VAR) is a multivariate forecasting technique that is used when two or more time-series are influencing each other. It is considered as an autoregressive model because, each variable in the observed feature space is modeled as a function of it's past values. VAR is used when there is a **bi-directional relationship** present in the observed dataset. Using this technique we can leverage the time-delayed relationship between features to forecast. The main two criterion for a VAR model is:

* There should be atleast two time-series
* The time-series should influence each other

In order to correctly identify features that influence each other, one need to perform causality and cointegration tests present in the **multivariate EDA** section.

**NOTE:** In order to avoid **n-th leading minor of the array is not positive definite"** error, user must make sure -
* The dataset is not too small
* The lag order used is not too large
* The data is stationary

Following code chunk takes the time-series data and trains the VAR model on it.

# Build models for all the panels
track_cell("VAR model building", flag)
dict_var  = {} # Dictionary to store the model objects and results
coefficient_table_var = pd.DataFrame()

if type(endog_variables) == str or len(endog_variables) == 1:
    display(Markdown('__VAR requires 2 endog variable__! Hence cannot be executed.'))
else:
    global error_table,valid_mape_table,model_var
    if data_type.lower()=='panel':
        ## Run model for each panel
        for id_col in tqdm(selected_panels):
            valid_grouped = valid.groupby(panel_col).get_group(id_col)
            train_grouped = train.groupby(panel_col).get_group(id_col)

            model_var = VARMAX(endog= train_grouped[endog_variables], 
                               exog= train_grouped[external_variables],
                               order = (2,0), trend = 'n')
            try:
                fit_var = model_var.fit()
            except:
                continue
            else:
                n_periods_ahead = len(valid_grouped)
                fitted_var = fit_var.forecast(train_grouped.shape[0], 
                                              exog = train_grouped[external_variables])
                forecasted_var = fit_var.forecast(n_periods_ahead, 
                                                  exog = valid_grouped[external_variables])

                error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],forecasted_var[target_var],id_col,'VAR','Validation')
                error_table = error_table.append(pd.DataFrame(error_local))
                valid_mape_table = valid_mape_table.append(error_local_mape)
                error_local,error_local_mape = get_error_metrics(train_grouped[target_var],fit_var.fittedvalues[target_var],id_col,'VAR','Train')
                error_table = error_table.append(pd.DataFrame(error_local))
                error_table.drop_duplicates(keep='last',inplace=True)
                valid_mape_table = valid_mape_table.append(error_local_mape)            
                dict_var.update({id_col:{'Model':fit_var,
                                         'Fitted': fitted_var,
                                        'Forecast':forecasted_var}})
                err_series = fit_var.params - fit_var.conf_int()[0]
                coef_df = pd.DataFrame({'coef': fit_var.params.values[0:],'err': err_series.values[0:],
                                        'varname': err_series.index.values[0:], 'panelID': id_col})
                coefficient_table_var = coefficient_table_var.append(coef_df)
        clear_output()
        display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">VARMAX models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_var))))
    else:
        display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
        model_var = VARMAX(endog= train[endog_variables], exog= train[external_variables],
                           order = (2,0), trend = 'c')
        fit_var = model_var.fit()
        n_periods_ahead = len(valid)
        fitted_var = fit_var.forecast(train.shape[0], exog = train[external_variables])
        forecasted_var = fit_var.forecast(n_periods_ahead, exog = valid[external_variables])

        error_local,error_local_mape = get_error_metrics(valid[target_var],forecasted_var[target_var],'NA','VAR','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train[target_var],fit_var.fittedvalues[target_var],'NA','VAR','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)
        dict_var.update({'Model':fit_var, 'Fitted':fitted_var, 'Forecast':forecasted_var})
        clear_output()
dict_models.update({'VAR':dict_var})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients for the selected external variables

if data_type.lower()=='panel':
    pid_ = random.choice(list(dict_var.keys()))
    if dict_var.get(pid_).get('Model') == None:
        display(Markdown('No summary generated!'))
    else:
        display(Markdown("Model Summary for panel ID: __{}__".format(pid_)))
        display(dict_var[pid_]['Model'].summary())
else:
    if dict_var.get('Model') == None:
        display(Markdown('No summary generated!'))
    else:
        display(dict_var['Model'].summary())

### Model fit on the entire dataset available [Naive Model]:

The model is also trained on the entire dataset to serve as a basis for whether a panel level or naive model would be a better fit for each panel

if data_type.lower() == 'panel':
    if type(endog_variables) == str or len(endog_variables) == 1:
        display(Markdown('__VAR requires 2 endog variable__! Hence cannot be executed.'))
    else:
        display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
        model_var = VARMAX(endog= train[endog_variables], exog= train[external_variables],
                           order = (2,0), trend = 'n')
        fit_var = model_var.fit()
        naive_coef = fit_var.params
        clear_output()
        display(fit_var.summary())
else:
    print(colored('You selected Non-Panel data! Hence this is not applicable.','magenta',attrs=['bold']))

### Coefficients plot
*  These plots are helpful in estimating the need for panel-wise models

If the coefficents of the panel wise models allign with the dotted line <b>coefficent of naive model</b>, the naive model can be used to model all panels
If any of the panel coefficents is distant from the dotted line <b>coefficent of naive model</b>, that panel requires it’s own model

if data_type.lower()=='panel':
    pid_ = random.choice(list(dict_var.keys()))
    if dict_var.get(pid_).get('Model') == None:
        display(Markdown('No plot generated!'))
    else:
        title_list = []
        variable_list = list(naive_coef.index)
        for i in variable_list:
            x = '{}<br>Naive value = {}'.format(i,round(naive_coef[i],4))
            title_list.append(x)

        title_tuple  = tuple(title_list)
        naive_coef = naive_coef.round(3)
        fig = make_subplots(rows=1, cols=len(variable_list), subplot_titles=title_tuple)
        for j in variable_list:
            df_subset = coefficient_table_var[coefficient_table_var['varname']==j]
            df_subset = df_subset.reset_index(drop = True)
            i = variable_list.index(j)+1
            fig = coef_plot(df_subset,i=i,val = naive_coef[j],l=len(variable_list))
        fig.show(config={'displaylogo': False})
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

### Model Estimation
Model metrics for training and validation data

if type(endog_variables) == str or len(endog_variables) == 1:
    display(Markdown('__VAR requires 2 endog variable__! Hence cannot be executed.'))
else:
    model_est(error_table,'VAR')

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

if type(endog_variables) == str or len(endog_variables) == 1:
    display(Markdown('__VAR requires 2 endog variable__! Hence cannot be executed.'))
else:
    fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
    if data_type.lower() == "panel":
        display(Markdown('__Processing...__'))
        processed_panels = dict_var.keys()
        for i,id_col in enumerate(tqdm(processed_panels)):
            visible = True if id_col == list(processed_panels)[0] else False
            fig = actual_vs_pred(vis = visible,method_name='VAR',
                                 train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                                 valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                                 fc_train= dict_var[id_col]['Model'].fittedvalues[target_var].reset_index(drop = True),
                                 fc_valid= dict_var[id_col]['Forecast'][target_var].reset_index(drop = True),
                                 target_var= target_var, panel = id_col, i= i)
        clear_output()
        fit_tabs(processed_panels,'VAR')
    else:
        fig = actual_vs_pred(method_name='VAR', vis=True,
                          train_data= train, valid_data= valid, target_var= target_var,
                          fc_train= dict_var['Model'].fittedvalues[target_var],
                          fc_valid= dict_var['Forecast'][target_var])
    clear_output()
    fig.show(config={'displaylogo': False})

## Support Vector Regression

Support Vector Regression or Support Vector Machines in general are perhaps one of the most popular and traditional
supervised machine learning approaches since 1990s. One of the most significance difference of linear regression from 
SVR is that instead of minimizing the error rate, we try to optimize the decision boundary line, so that the error is within a certain threshold.
SVR has the capability to handle linear as we as non-linear data by using different sort of mapping function used to project lower dimensional 
data into a higher dimension, which are also called kernels.
For further information visit [this](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) link.

Following code chunk takes the time-series data and trains the SVR model on it.

track_cell("SVR model building", flag)
model_svr = SVR(kernel= 'linear')
dict_svr  = {} # Dictionary to store the model objects and results

global error_table,valid_mape_table
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_svr.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'SVR',col=id_col)
            dict_svr.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,
                                    'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,
                                    'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">SVR models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_svr))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_svr.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'SVR')
    dict_svr.update({'Model':fit_model,
                    'Forecast':forecasted_model,
                    'Fitted': fitted_model,
                    'Residuals_train': residuals_train,
                    'Residuals_validation':residuals_validation})
    clear_output()
dict_models.update({'SVR':dict_svr})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],
                                            round(dict_svr[pid]['Model'].coef_.tolist()[0][i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_svr[pid]['Model'].intercept_.tolist()[0], 4))))
else:
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],
                                            round(dict_svr['Model'].coef_.tolist()[0][i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_svr['Model'].intercept_.tolist()[0], 4))))

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'SVR')

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_svr.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name='SVR',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True), 
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= pd.Series(dict_svr[id_col]['Fitted']),
                             fc_valid= pd.Series(dict_svr[id_col]['Forecast']),
                             target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'SVR')
else:
    fig = actual_vs_pred(method_name='SVR', vis=True,
                         train_data= train, valid_data= valid, target_var= target_var,
                         fc_train= pd.Series(dict_svr['Fitted']),
                         fc_valid= pd.Series(dict_svr['Forecast']))
clear_output()
fig.show(config={'displaylogo': False})

## Decision Tree

Decision Tree is a technique that is more widely used for classification but can also be applied as a regression model. It is a binary recursive technique. Classification and Regression Trees (CART) split attributes based on values that minimize a loss function, such as sum of squared errors to fine tune results.

Following code chunk takes the time-series data and trains the decision tree model on it.

dict_dt  = {}
model_dt = DecisionTreeRegressor(criterion='mse',max_depth=2,min_samples_split=3)
model_name = 'Decision Tree'

global error_table,valid_mape_table
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_dt.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'Decision Tree',col=id_col)
            dict_dt.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,
                                    'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,
                                    'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">Decision Tree Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_dt))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_dt.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'Decision Tree')
    dict_dt.update({'Model':fit_model,
                    'Forecast':forecasted_model,
                    'Fitted': fitted_model,
                    'Residuals_train': residuals_train,
                    'Residuals_validation':residuals_validation})
    clear_output()
dict_models.update({model_name:dict_dt})
print("Model trained successfully")

### Tree Plot

plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots(figsize=(10, 10))
if data_type.lower()=='panel':
    plot_tree(dict_dt[pid]['Model'], rotate=True, filled = True, ax=ax, max_depth=3,
              feature_names= external_variables)
else:
    plot_tree(dict_dt['Model'], rotate=True, filled = True, max_depth=3, ax=ax,
              feature_names= external_variables)  #rounded
plt.show()

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'Decision Tree')      

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_dt.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name= 'Decision Tree',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= pd.Series(dict_dt[id_col]['Fitted']),
                             fc_valid= pd.Series(dict_dt[id_col]['Forecast']),
                             target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'Decision Tree')
else:
    fig = actual_vs_pred(method_name= 'Decision Tree', vis=True,
                         train_data= train, valid_data= valid.reset_index(drop = True), 
                         target_var= target_var,
                         fc_train= pd.Series(dict_dt['Fitted']),
                         fc_valid= pd.Series(dict_dt['Forecast']))

clear_output()
fig.show(config={'displaylogo': False})

## Random Forest

Random forests are an ensemble learning method that operate by constructing a multitude of decision trees at training time and outputting mean prediction of the individual trees. Random forests correct for decision trees’ habit of overfitting to their training set.

Following code chunk takes the time-series data and trains the random-forest model on it.

### Model Summary

track_cell("RF model building", flag)
dict_rf  = {}
model_rf = RandomForestRegressor()


global error_table,valid_mape_table
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_rf.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'Random Forest',col=id_col)
            dict_rf.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,
                                    'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,
                                    'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">Random Forest Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_rf))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_rf.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'Random Forest')
    dict_rf.update({'Model':fit_model,
                    'Forecast':forecasted_model,
                    'Fitted': fitted_model,
                    'Residuals_train': residuals_train,
                    'Residuals_validation':residuals_validation})
    clear_output()
dict_models.update({'Random Forest':dict_rf})
print("Model trained successfully")        

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'Random Forest')     

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_rf.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name= 'Random Forest',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True), 
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= pd.Series(dict_rf[id_col]['Fitted']),
                             fc_valid= pd.Series(dict_rf[id_col]['Forecast']),
                             target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'Random Forest')
else:
    fig = actual_vs_pred(method_name= 'Random Forest', vis=True,
                         train_data= train, valid_data= valid.reset_index(drop = True), 
                         target_var= target_var,
                         fc_train= pd.Series(dict_rf['Fitted']),
                         fc_valid= pd.Series(dict_rf['Forecast']))
clear_output()
fig.show(config={'displaylogo': False})

## XGBoost


The __eXtreme Gradient Boosting (XGBoost)__ model is an implementation of the gradient boosting framework. Gradient Boosting algorithm is a machine learning technique used for building predictive tree-based models. Boosting is an ensemble technique in which new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. The ensemble technique uses the tree ensemble model which is a set of classification and regression trees (CART). The ensemble approach is used because a single CART, usually, does not have a strong predictive power. By using a set of CART (i.e. a tree ensemble model) a sum of the predictions of multiple trees is considered.

### Model Summary
Following code chunk takes the time-series data and trains the xgboost model on it to display the model coefficients

dict_xgb={}
model_name = 'XG Boost'
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, 
                             learning_rate = 0.6, max_depth = 2, alpha = 10, 
                             n_estimators = 10, booster='gbtree', early_stopping_rounds = 10)

global error_table,valid_mape_table
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_xgb.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'XG Boost',col=id_col)
            dict_xgb.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,
                                    'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,
                                    'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">XGBoost Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_xgb))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_xgb.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'XG Boost')
    dict_xgb.update({'Model':fit_model,
                    'Forecast':forecasted_model,
                    'Fitted': fitted_model,
                    'Residuals_train': residuals_train,
                    'Residuals_validation':residuals_validation})
    clear_output()        
dict_models.update({model_name:dict_xgb})
print("Model trained successfully")

### Model Estimation
Model metrics for training and validation data

model_est(error_table,'XG Boost')       

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_xgb.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name='XG Boost',
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= pd.Series(dict_xgb[id_col]['Fitted']),
                             fc_valid= pd.Series(dict_xgb[id_col]['Forecast']),
                             target_var= target_var, panel = id_col, i=i)
    clear_output()
    fit_tabs(processed_panels,'XG Boost')
else:
    fig = actual_vs_pred(method_name='XG Boost', vis=True,
                         train_data= train, valid_data= valid.reset_index(drop = True), 
                         target_var= target_var,
                         fc_train= pd.Series(dict_xgb['Fitted']),
                         fc_valid= pd.Series(dict_xgb['Forecast']))
clear_output()
fig.show(config={'displaylogo': False})

## Generalized Additive Model Regression (GAM)<a name = 'gam'></a>

A generalized additive model (GAM) is a linear model in which the predicted values depend linearly on the smooth functions of the provided regressors.

The formula supplied to a GAM model is similar to that of a linear regression model with the added functionality of being able to smooth regressors using s().

########################################################################################
################################## User Input Needed ###################################
########################################################################################
# variables to pass as linear to the GAM formula
lin_col= ['max_rooms_capacity', 'avgdailyrate', 'loyalty_pct'] # options - for panel ['rms_avail_qty'] 
# variables to pass as spline to the GAM formula
spl_col= ['percentbusiness', 'percentgovtnights'] # options - for panel ['age', 'mpi'] 
# Note - linear and spline columns should have no intersection 
########################################################################################
dict_gam  = {}
train_multi = train[external_variables]
train_multi = train_multi.dropna()
column_names=train_multi.columns
spline = [train_multi.columns.get_loc(c) for c in spl_col if c in train_multi]
linear = [train_multi.columns.get_loc(c) for c in lin_col if c in train_multi]
list1 = ['pygam.l({})'.format(i) for i in linear]
list2 = ['pygam.s({})'.format(i) for i in spline]
argument_to_pass = '{}+{}'.format('+'.join(list1), '+'.join(list2))
model_gam = pygam.GAM(eval(argument_to_pass)) # model obj initialized 
model_name = 'GAM'

global error_table,valid_mape_table
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_gam.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'GAM',col=id_col)
            dict_gam.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,
                                    'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,
                                    'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">GAM models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_gam))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_gam.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'GAM')
    dict_gam.update({'Model': fit_model,
                    'Forecast': forecasted_model,
                    'Fitted': fitted_model,
                    'Residuals_train': residuals_train,
                    'Residuals_validation': residuals_validation})
    clear_output()
dict_models.update({model_name:dict_gam})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(dict_gam[pid]['Model'].summary())
else:
    display(dict_gam['Model'].summary())       ## Display model summary   

### Partial Dependence Plot

The following section shows the plots controlling the selected smoothing parameter for the GAM model.Smoothing parameter allows us to explicitly balance the bias/variance trade-off; smoother curves have more bias (in-sample error), but also less variance. Curves with less variance tend to make more sense and validate better in out-of-sample tests. These plots also are directly controlled by the degrees of freedom of the smoothing parameter. Higher the degree, more flexible the curve.

titles = spl_col.copy()
fig = go.Figure()
for i,each_col in enumerate(tqdm(titles)):
    visible = True if each_col == titles[0] else False
    if data_type.lower()=='panel':
        x_grid = dict_gam[pid]['Model'].generate_X_grid(term=i)
        X = x_grid[:, i]
        Y = dict_gam[pid]['Model'].partial_dependence(term=i, X=x_grid).tolist()
        intervals = dict_gam[pid]['Model'].partial_dependence(term=i, X=x_grid, width=.95)[1].tolist()
        lower_interval = [sublist[0] for sublist in intervals]
        upper_interval = [sublist[1] for sublist in intervals]
    else:
        x_grid = dict_gam['Model'].generate_X_grid(term=i)
        X = x_grid[:, i]
        Y = dict_gam['Model'].partial_dependence(term=i, X=x_grid).tolist()
        intervals = dict_gam['Model'].partial_dependence(term=i, X=x_grid, width=.95)[1].tolist()
        lower_interval = [sublist[0] for sublist in intervals]
        upper_interval = [sublist[1] for sublist in intervals]
    fig.add_trace(go.Scatter(x= X,y= Y,
                             mode = 'lines',name = 'y',opacity= 0.8,
                             showlegend= False,visible = visible,legendgroup='y'))
    fig.add_trace(go.Scatter(x= X,y= lower_interval,
                             mode = 'lines+markers',line_color = 'red',name = 'conf_int_lower',
                             opacity= 0.5,showlegend= False,visible = visible,
                             legendgroup='conf_int_lower'))
    fig.add_trace(go.Scatter(x= X,y= upper_interval,
                             mode = 'lines+markers',name = 'conf_int_upper',line_color = 'red',
                             opacity= 0.5,showlegend= False,visible = visible,legendgroup='conf_int_upper'))

tab_dict_list = []
for each_col in titles:
    vis_check = [[True]*3 if i==each_col else [False]*3 for i in titles]
    vis_check_flat = [i for sublist in vis_check for i in sublist]
    tab_dict_list.append(dict(args = [{"visible": vis_check_flat},
                                      {"title": "Partial Dependence Plot  : {}".format(each_col)}],
                                label=each_col, method="update"))
    fig.update_layout(updatemenus=[dict(buttons=list(tab_dict_list),
                                    direction="right",
                                    x=0, xanchor="left", y=1.11, yanchor="top")],
                      showlegend=False, title_x = 0.5)
clear_output()
fig.show(config={'displaylogo': False})

### Model Estimation
Model metrics for training and validation data

model_est(error_table,model_name)        

### Goodness of Fit
Following code chunk displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_gam.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name= model_name,
                             train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             valid_data= valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                             fc_train= pd.Series(dict_gam[id_col]['Fitted']),
                             fc_valid= pd.Series(dict_gam[id_col]['Forecast']),
                             target_var= target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,model_name)
else:
    fig = actual_vs_pred(method_name=model_name, vis=True,
                         train_data= train, valid_data= valid.reset_index(drop = True), 
                         target_var= target_var,
                         fc_train= pd.Series(dict_gam['Fitted']),
                         fc_valid= pd.Series(dict_gam['Forecast']))
clear_output()
fig.show(config={'displaylogo': False})

## Quantile Regression

Quantile regression is an extension of linear regression that is used when the conditions of linear regression are not met (i.e., linearity, homoscedasticity, independence, or normality). Instead of optimizing (minimizing) the cost function best to estimate the conditional mean, Quantile Regression estimates the conditional quantile (eg. median, first quantile, third quantile or any nth quantile) of the target. This method uses LAD or Least Absolute Deviation along with linear programming (eg. simplex method) to optimize the error as oppose to OLS in simple linear regression.


dict_quant  = {}
coefficient_table_quant = pd.DataFrame()
model_name = 'QUANTILE'

global error_table,valid_mape_table,forecasted_model
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)

        try:
            model_quant = smf.quantreg(formula, data=train_grouped).fit(q=0.4)
        except:
            print(colored('You have multicolinearity in the data for panel {}!'.format(id_col),
                          'red'))
        else:
            forecasted_model = model_quant.predict(valid_grouped)
            error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],forecasted_model,
                                                     id_col,'QUANTILE','Validation')
            error_table = error_table.append(pd.DataFrame(error_local))
            valid_mape_table = valid_mape_table.append(error_local_mape)
            error_local,error_local_mape = get_error_metrics(train_grouped[target_var],model_quant.fittedvalues,
                                                             id_col,'QUANTILE','Train')
            error_table = error_table.append(pd.DataFrame(error_local))
            error_table.drop_duplicates(keep='last',inplace=True)
            valid_mape_table = valid_mape_table.append(error_local_mape)
            
            dict_quant.update({id_col:{'Model':model_quant,'Fitted': model_quant.fittedvalues,
                                       'Forecast':forecasted_model}})

            err_series = model_quant.params - model_quant.conf_int()[0]
            coef_df = pd.DataFrame({'coef': model_quant.params.values[1:],
                                    'err': err_series.values[1:],
                                    'varname': err_series.index.values[1:], 
                                    'panelID': id_col})
            coefficient_table_quant = coefficient_table_quant.append(coef_df)
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">Quantile Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_quant))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    try:
        model_quant = smf.quantreg(formula,data=train).fit(q=0.4)
    except:
        clear_output()
        print(colored('You have multicolinearity in the data!','red'))
    else:
        forecasted_model = model_quant.predict(valid)
        error_local,error_local_mape = get_error_metrics(valid[target_var],forecasted_model,
                                                 'NA','QUANTILE','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train[target_var],model_quant.fittedvalues,
                                                         'NA','QUANTILE','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)
        dict_quant.update({'Model':model_quant,
                           'Fitted': model_quant.fittedvalues,
                           'Forecast':forecasted_model})
        clear_output()
dict_models.update({model_name:dict_quant})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients for the selected external variables

if dict_quant != {}:
    if data_type.lower()=='panel':
        display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
        print(dict_quant[pid]['Model'].summary())
    else:
        display(dict_quant['Model'].summary())       ## Display model summary
else:
    print(colored('Model did not execute due to multicolinearity in the data!','red'))

### Model fit on the entire dataset available [Naive Model]:

The model is also trained on the entire dataset to serve as a basis for whether a panel level or naive model would be a better fit for each panel

if data_type.lower() == 'panel':
    model_quant = smf.quantreg(formula, data=train).fit()
    naive_coef = model_quant.params
    display(model_quant.summary())
else:
    print(colored('You selected Non-Panel data! Hence this is not applicable.','magenta',attrs=['bold']))

### Coefficients plot
*  These plots are helpful in estimating the need for panel-wise models

If the coefficents of the panel wise models allign with the dotted line <b>coefficent of naive model</b>, the naive model can be used to model all panels
If any of the panel coefficents is distant from the dotted line <b>coefficent of naive model</b>, that panel requires it’s own model

if data_type.lower()=='panel':
    if dict_quant != {}:
        title_list = []
        variable_list = list(naive_coef.index[1:])

        for i in variable_list:
            x = '{}, Naive value =  {}'.format(i,naive_coef[i])
            title_list.append(x)

        title_tuple  = tuple(title_list)
        naive_coef = naive_coef.round(3)
        fig = make_subplots(rows=1, cols=len(variable_list), subplot_titles=title_tuple)
        for j in variable_list:
            df_subset = coefficient_table_quant[coefficient_table_quant['varname']==j]
            df_subset = df_subset.reset_index(drop = True)
            i = variable_list.index(j)+1
            fig = coef_plot(df_subset,i=i,val = naive_coef[i],l=len(variable_list))

        fig.show(config={'displaylogo': False})
    else:
        print(colored('Model did not execute due to multicolinearity in the data!','red'))
else:
    print('\033[35;1m This section is not applicable for non-panel data. \033[0m')

### Model Estimation
Model metrics for training and validation data

if dict_quant != {}:
    model_est(error_table,model_name)
else:
    print(colored('Model did not execute due to multicolinearity in the data!','red'))

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

if dict_quant != {}:
    fig = make_subplots(rows=4, cols=1, 
                        subplot_titles=['Actual vs Predicted','Residual vs Time',
                                        'Residual ACF','Residual PACF'],
                        vertical_spacing = 0.1)
    if data_type.lower() == "panel":
        display(Markdown('__Processing...__'))
        processed_panels = dict_quant.keys()
        for i,id_col in enumerate(tqdm(processed_panels)):
            visible = True if id_col == list(processed_panels)[0] else False
            fig = actual_vs_pred(vis = visible,method_name= model_name,
                                 train_data= train.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                                 valid_data = valid.groupby(panel_col).get_group(id_col).reset_index(drop = True),
                                 fc_train= dict_quant[id_col]['Model'].fittedvalues.reset_index(drop = True),
                                 fc_valid= dict_quant[id_col]['Forecast'].reset_index(drop = True),
                                 target_var= target_var, panel = id_col, i= i)
        clear_output()
        fit_tabs(processed_panels,model_name)
    else:
        fig = actual_vs_pred(method_name= model_name, vis=True,
                             train_data= train, valid_data= valid.reset_index(drop = True), 
                             target_var= target_var,
                             fc_train= dict_quant['Model'].fittedvalues.reset_index(drop = True),
                             fc_valid= dict_quant['Forecast'].reset_index(drop = True))
    clear_output()
    fig.show(config={'displaylogo': False})
else:
    print(colored('Model did not execute due to multicolinearity in the data!','red'))

## PLS

Partial least squares (PLS) regression is a technique that reduces the predictors to a smaller set of uncorrelated components and performs least squares regression on these components, instead of on the original data. PLS regression is especially useful when your predictors are highly collinear, or when you have more predictors than observations and ordinary least-squares regression either produces coefficients with high standard errors or fails completely. PLS does not assume that the predictors are fixed, unlike multiple regression. This means that the predictors can be measured with error, making PLS more robust to measurement uncertainty.

PLS regression is primarily used in the chemical, drug, food, and plastic industries. A common application is to model the relationship between spectral measurements (NIR, IR, UV), which include many variables that are often correlated with each other, and chemical composition or other physio-chemical properties. In PLS regression, the emphasis is on developing predictive models. Therefore, it is not usually used to screen out variables that are not useful in explaining the response.

Following code chunk takes the time-series data and trains the pls model on it.

dict_pls  = {}
model_pls = PLSRegression(n_components= len(external_variables))

global error_table,valid_mape_table,fit_pls
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)

        fit_pls = model_pls.fit(X = train_grouped[external_variables], 
                                Y = train_grouped[target_var])
        fitted_pls = fit_pls.predict(train_grouped[external_variables])

        fitted_pls_unlisted = []
        for sublist in fitted_pls:
            for item in sublist:
                fitted_pls_unlisted.append(item)

        fitted_pls = map(lambda x: x[0], fitted_pls)
        fitted_pls_sr = pd.Series(fitted_pls)
        residuals_train = train_grouped[target_var] - fitted_pls_unlisted
        forecasted_pls = fit_pls.predict(valid_grouped[external_variables])
        forecasted_pls_unlisted = []
        for sublist in forecasted_pls:
            for item in sublist:
                forecasted_pls_unlisted.append(item)
        residuals_val = valid_grouped[target_var] - forecasted_pls_unlisted

        error_local,error_local_mape = get_error_metrics(valid_grouped[target_var],forecasted_pls_unlisted,id_col,'PLS','Validation')
        error_table = error_table.append(pd.DataFrame(error_local))
        valid_mape_table = valid_mape_table.append(error_local_mape)
        error_local,error_local_mape = get_error_metrics(train_grouped[target_var],fitted_pls_unlisted,id_col,'PLS','Train')
        error_table = error_table.append(pd.DataFrame(error_local))
        error_table.drop_duplicates(keep='last',inplace=True)
        valid_mape_table = valid_mape_table.append(error_local_mape)
        dict_pls.update({id_col:{'Model':fit_pls,
                                'Forecast':forecasted_pls_unlisted,
                                'Fitted': fitted_pls_sr,
                                'Residuals_train': residuals_train,
                                'Residuals_validation':residuals_val}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">PLS Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_pls))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_pls = model_pls.fit(X = train[external_variables], 
                            Y = train[target_var])
    fitted_pls = fit_pls.predict(train[external_variables])
    fitted_pls_unlisted = []
    for sublist in fitted_pls:
        for item in sublist:
            fitted_pls_unlisted.append(item)

    residuals_train = train[target_var] - fitted_pls_unlisted
    forecasted_pls = fit_pls.predict(valid[external_variables])

    forecasted_pls_unlisted = []
    for sublist in forecasted_pls:
        for item in sublist:
            forecasted_pls_unlisted.append(item)
    residuals_val = valid[target_var] - forecasted_pls_unlisted

    error_local,error_local_mape = get_error_metrics(valid[target_var],forecasted_pls_unlisted,'NA','PLS','Validation')
    error_table = error_table.append(pd.DataFrame(error_local))
    valid_mape_table = valid_mape_table.append(error_local_mape)
    error_local,error_local_mape = get_error_metrics(train[target_var],fitted_pls_unlisted,'NA','PLS','Train')
    error_table = error_table.append(pd.DataFrame(error_local))
    error_table.drop_duplicates(keep='last',inplace=True)
    valid_mape_table = valid_mape_table.append(error_local_mape)
    dict_pls.update({'Model':fit_pls,'Forecast':forecasted_pls_unlisted,
                    'Fitted': fitted_pls,'Residuals_train': residuals_train,'Residuals_validation':residuals_val})
    clear_output()
print("Model trained successfully")
dict_models.update({'PLS':dict_pls})

### Model Summary
Following code chunk displays the model coefficients for the selected external variables

 if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],round(dict_pls[pid]['Model'].coef_.tolist()[i][0], 4))))
else:
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],round(dict_pls['Model'].coef_.tolist()[i][0], 4))))  

### Model Estimation
Model metrics for training and Validation data

model_est(error_table,'PLS')

### Goodness of Fit

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_pls.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible, method_name = 'PLS',
                             train_data = train.groupby(panel_col).get_group(id_col),
                             valid_data = valid.groupby(panel_col).get_group(id_col),
                             fc_train = dict_pls[id_col]['Fitted'],
                             fc_valid = dict_pls[id_col]['Forecast'],
                             target_var = target_var, panel = id_col, i= i)
    clear_output()
    fit_tabs(processed_panels,'PLS')
else:
    fig = actual_vs_pred(method_name='PLS', vis=True,
                         train_data= train, valid_data = valid, 
                         target_var= target_var,
                         fc_train= pd.Series(dict_pls['Fitted'].flatten()),
                         fc_valid= dict_pls['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})
        

## Ridge regression<a name = 'ridge'>  </a>

Ridge regression is an extension of linear regression where the loss function is modified to minimize the complexity of the model. This modification is done by adding a penalty parameter that is equivalent to the square of the magnitude of the coefficients.
 
The penalty term regularizes the coefficients such that if the coefficients take large values the optimization function is penalized. So, ridge regression shrinks the coefficients and it helps to reduce the model complexity and multi-collinearity

To know in details about hyperparameters used in this model, refer to this [link](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

Following code chunk takes the time-series data and trains the rigde regression model on it.

dict_ridge  = {}
global error_table,valid_mape_table
model_ridge = Ridge(alpha=0)
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization, 
# coefficients are barely restricted and in this case linear and ridge regression resembles

if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_ridge.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'Ridge',col=id_col)
            dict_ridge.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,
                                    'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,
                                    'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">Ridge Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_ridge))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_ridge.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'Ridge')
    dict_ridge.update({'Model':fit_model,
                    'Forecast':forecasted_model,
                    'Fitted': fitted_model,
                    'Residuals_train': residuals_train,
                    'Residuals_validation':residuals_validation})
    clear_output()
        
dict_models.update({'Ridge':dict_ridge})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients for the selected external variables

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],round(dict_ridge[pid]['Model'].coef_.tolist()[i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_ridge[pid]['Model'].intercept_, 4))))
else:
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],
                                            round(dict_ridge['Model'].coef_.tolist()[i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_ridge['Model'].intercept_, 4))))

### Model Estimation
Model metrics for training and Validation data

model_est(error_table,"Ridge")

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_ridge.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name= "Ridge",
                              train_data= train.groupby(panel_col).get_group(id_col),
                              target_var= target_var,
                              valid_data = valid.groupby(panel_col).get_group(id_col),
                              fc_train= dict_ridge[id_col]['Fitted'],
                              fc_valid= dict_ridge[id_col]['Forecast'],
                              panel = id_col,i= i)
    clear_output()
    fit_tabs(processed_panels,"Ridge")
else:
    fig = actual_vs_pred(method_name= "Ridge", vis=True,
                      train_data= train,target_var= target_var,valid_data = valid,
                      fc_train= dict_ridge['Fitted'],fc_valid= dict_ridge['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

## LASSO Regression


Least Absolute Shrinkage and Selection Operator (LASSO) regression is a technique that works on the principle of shrinkage. Data values are shrunk to some type of a central point like the mean. The models created are simple and sparse, making them useful for data with high multicollinearity.

LASSO performs regularization, where penalties are given based on the absolute value of the magnitude of the coefficient of the independent variable. Larger the value, closer the coefficients are to zero, creating a simple and effective model.

The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters). This particular type of regression is well-suited for models showing high levels of muticollinearity or when you want to automate certain parts of model selection, like variable selection/parameter elimination. Lasso regression performs L1 regularization, which adds a penalty equal to the absolute value of the magnitude of coefficients. This type of regularization can result in sparse models with few coefficients; Some coefficients can become zero and eliminated from the model. Larger penalties result in coefficient values closer to zero, which is the ideal for producing simpler models. On the other hand, L2 regularization (e.g. Ridge regression) doesn’t result in elimination of coefficients or sparse models. This makes the Lasso far easier to interpret than the Ridge.

When alpha = 1, then the model is called lasso regularization model.

To know in details about hyperparameters used in this model, refer to this [link](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).

Following code chunk takes the time-series data and trains the lasso model on it.

dict_lasso  = {} # store model training output
global error_table,valid_mape_table
model_lasso = Lasso(alpha=0.5)
if data_type.lower()=='panel':
    ## Run model for each panel
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_lasso.fit(X = train_grouped[external_variables], 
                                      y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'LASSO',col=id_col)
            dict_lasso.update({id_col:{'Model':fit_model,
                                    'Forecast':forecasted_model,'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">LASSO Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_lasso))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_lasso.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'LASSO')
    dict_lasso.update({'Model':fit_model, 'Forecast':forecasted_model,
                       'Fitted': fitted_model,'Residuals_train': residuals_train,
                       'Residuals_validation':residuals_validation})
    clear_output()
    
dict_models.update({'LASSO':dict_lasso})
print("Model trained successfully")

### Model Summary
Following code chunk displays the model coefficients for the selected external variables

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],
                                            round(dict_lasso[pid]['Model'].coef_.tolist()[i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_lasso[pid]['Model'].intercept_, 4))))
else:
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],
                                            round(dict_lasso['Model'].coef_.tolist()[i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_lasso['Model'].intercept_, 4))))

### Model Estimation
Model metrics for training and Validation data

model_est(error_table,"LASSO")

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time',
                                    'Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_lasso.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name="LASSO",
                              train_data= train.groupby(panel_col).get_group(id_col),
                              target_var= target_var,
                              valid_data = valid.groupby(panel_col).get_group(id_col),
                              fc_train= dict_lasso[id_col]['Fitted'],
                              fc_valid= dict_lasso[id_col]['Forecast'],
                              panel = id_col,i= i)
    clear_output()
    fit_tabs(processed_panels,"LASSO")
else:
    fig = actual_vs_pred(method_name="LASSO", vis=True,
                         train_data= train,target_var= target_var,valid_data = valid,
                         fc_train= dict_lasso['Fitted'],fc_valid= dict_lasso['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

## Elastic Net 

Elastic net is a combination of ridge and lasso regularization which in theory, overcomes the limitations of both. What this means is that with elastic net the algorithm can remove weak variables altogether as with lasso or to reduce them to close to zero as with ridge. In elastic net, along with the λ hyper-parameter, alpha also is tuned within a grid to get the optimal combination for (α Lasso + 1−α Ridge).

In this case (0 < α < 1).



To know in details about hyperparameters used in this model, refer to this [link](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).

Following code chunk takes the time-series data and trains the elastic net model on it.

track_cell("ElasticNet model building", flag) # tracking 
dict_enet  = {} # store model training output
model_enet = ElasticNet(alpha = 0.01) # initialize model object

global error_table,valid_mape_table
if data_type.lower()=='panel':
    for id_col in tqdm(selected_panels):
        valid_grouped = valid.groupby(panel_col).get_group(id_col)
        train_grouped = train.groupby(panel_col).get_group(id_col)
        try:
            fit_model = model_enet.fit(X = train_grouped[external_variables],y = train_grouped[target_var])
        except:
            continue
        else:
            run_model(fit_model,train_grouped,valid_grouped,'ENET',col=id_col)
            dict_enet.update({id_col:{'Model':fit_model,'Forecast':forecasted_model,'Fitted': fitted_model,
                                    'Residuals_train': residuals_train,'Residuals_validation':residuals_validation}})
    clear_output()
    display(Markdown('<span style="color:darkgreen; font-style: bold; font-size: 15px">Elastic Net Regression models for <b>{}</b> panels have been built and stored in a dictionary</span>'.format(len(dict_enet))))
else:
    display(Markdown('<div><div class="loader"></div><h2> &nbsp; Processing...</h2></div>'))
    fit_model = model_enet.fit(X = train[external_variables], y = train[target_var])
    run_model(fit_model,train,valid,'ENET')
    dict_enet.update({'Model':fit_model,'Forecast':forecasted_model,'Fitted': fitted_model,
                    'Residuals_train': residuals_train,'Residuals_validation':residuals_validation})
    clear_output()
dict_models.update({'ENET':dict_enet})
print("Model trained successfully")

### Model Summary

Following code chunk displays the model coefficients for the selected external variables

if data_type.lower()=='panel':
    display(Markdown("Model Summary for panel ID: __{}__".format(pid)))
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],round(dict_enet[pid]['Model'].coef_.tolist()[i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_enet[pid]['Model'].intercept_, 4))))
else:
    display(Markdown("__Coefficients:__"))
    for i in range(len(external_variables)):
        display(Markdown("_{}_ : {}".format(external_variables[i],round(dict_enet['Model'].coef_.tolist()[i], 4))))
    display(Markdown("`Intercept`: {}".format(round(dict_enet['Model'].intercept_, 4))))

### Model Estimation
Model metrics for training and Validation data

model_est(error_table,'ENET')

### Goodness of Fit
Following code chunks displays the residual plots for the fitted model

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=['Actual vs Predicted','Residual vs Time','Residual ACF','Residual PACF'],
                    vertical_spacing = 0.1)
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    processed_panels = dict_enet.keys()
    for i,id_col in enumerate(tqdm(processed_panels)):
        visible = True if id_col == list(processed_panels)[0] else False
        fig = actual_vs_pred(vis = visible,method_name='ENET',
                             train_data= train.groupby(panel_col).get_group(id_col),
                             target_var= target_var,
                             valid_data = valid.groupby(panel_col).get_group(id_col),
                             fc_train= dict_enet[id_col]['Fitted'],
                             fc_valid= dict_enet[id_col]['Forecast'],
                             panel = id_col, i=i)
    clear_output()
    fit_tabs(processed_panels,'ENET')
else:
    fig = actual_vs_pred(method_name='ENET', vis=True,
                        train_data= train,target_var= target_var,valid_data = valid,
                        fc_train= dict_enet['Fitted'],fc_valid= dict_enet['Forecast'])
clear_output()
fig.show(config={'displaylogo': False})

# Model Comparison <a name ='model_compare'></a>

This section allows the user to compare models based on different evaluation metrics and understand which model is the best fit for the data.

## MAPE

## MAPE
## Creating horizontal bar charts to compare Model metrics
## ------------------------------------------------------------------------------------------------------------
track_cell("Model comparison MAPE", flag)

compare_df = pd.DataFrame(error_table.groupby(['model','datatype'],
                                              group_keys=False).apply(lambda x: x['MAPE'].mean()).reset_index())
compare_df.columns = ['model','datatype','values']
compare_df = compare_df.pivot(index='model', columns='datatype', 
                              values='values').reset_index()
compare_df = compare_df.reset_index(drop = True)
compare_df = compare_df.sort_values(by = 'Validation', ascending = True)
compare_df = compare_df.reset_index(drop = True)

fig = go.Figure(data=[go.Bar(name='Train', y=compare_df['model'], 
                             marker=dict(color='dodgerblue'),
                             x=compare_df['Train'], orientation = 'h'),
                      go.Bar(name='Validation', y=compare_df['model'], 
                             marker=dict(color='darkblue'),
                             x=compare_df['Validation'], orientation = 'h')])

fig.update_layout(legend=dict(orientation="h", yanchor="bottom", 
                              y=-0.35, xanchor="left", x=0), title='MAPE', title_x=0.5)
if data_type.lower()=='panel':
    fig.update_layout(barmode='group', title={"text": ("{} for each model (mean across panels)").format('MAPE')})

fig.show(config={'displaylogo': False})

## Creating box plots to compare Model metrics
## ------------------------------------------------------------------------------------------------------------
fig = px.box(error_table, x="model", y="MAPE", color="datatype")
display(Markdown("__Box plot__ of error metric : __MAPE__"))
fig.update_layout(legend=dict(orientation="h", 
                              yanchor="bottom", y=-0.35, xanchor="left", x=0))
fig.show(config={'displaylogo': False})

## SMAPE

## SMAPE
## Creating horizontucm_model.cycleal bar charts to compare Model metrics
## ------------------------------------------------------------------------------------------------------------
track_cell("Model Comparison SMAPE", flag)
compare_df = pd.DataFrame(error_table.groupby(['model','datatype'],
                                              group_keys=False).apply(lambda x: x['SMAPE'].mean()).reset_index())
compare_df.columns = ['model','datatype','values']
compare_df = compare_df.pivot(index='model', columns='datatype', 
                              values='values').reset_index()
compare_df = compare_df.reset_index(drop = True)
compare_df = compare_df.sort_values(by = 'Validation', ascending = True)
compare_df = compare_df.reset_index(drop = True)

fig = go.Figure(data=[go.Bar(name='Train', y=compare_df['model'], 
                             marker=dict(color='dodgerblue'),
                             x=compare_df['Train'], orientation = 'h'),
                      go.Bar(name='Validation', y=compare_df['model'], 
                             marker=dict(color='darkblue'),
                             x=compare_df['Validation'], orientation = 'h')])

fig.update_layout(legend=dict(orientation="h", yanchor="bottom", 
                              y=-0.35, xanchor="left", x=0), title='SMAPE', title_x=0.5)
if data_type.lower()=='panel':
    fig.update_layout(barmode='group', title={"text": ("{} for each model (mean across panels)").format('SMAPE')})

fig.show(config={'displaylogo': False})

## Creating box plots to compare Model metrics
## ------------------------------------------------------------------------------------------------------------
fig = px.box(error_table, x="model", y="SMAPE", color="datatype")
display(Markdown("__Box plot__ of error metric : __SMAPE__"))
fig.update_layout(legend=dict(orientation="h", 
                              yanchor="bottom", y=-0.35, xanchor="left", x=0))
fig.show(config={'displaylogo': False})

## MSE

## MSE
## Creating horizontal bar charts to compare Model metrics
track_cell("Model comparison MSE", flag)
        
compare_df = pd.DataFrame(error_table.groupby(['model','datatype'],
                                                  group_keys=False).apply(lambda x: x['MSE'].mean()).reset_index())
compare_df.columns = ['model','datatype','values']
compare_df = compare_df.pivot(index='model', columns='datatype', 
                              values='values').reset_index()
compare_df = compare_df.reset_index(drop = True)
compare_df = compare_df.sort_values(by = 'Validation', ascending = True)
compare_df = compare_df.reset_index(drop = True)

fig = go.Figure(data=[go.Bar(name='Train', y=compare_df['model'], 
                             marker=dict(color='dodgerblue'),
                             x=compare_df['Train'], orientation = 'h'),
                      go.Bar(name='Validation', y=compare_df['model'], 
                             marker=dict(color='darkblue'),
                             x=compare_df['Validation'], orientation = 'h')])

fig.update_layout(legend=dict(orientation="h", yanchor="bottom", 
                              y=-0.35, xanchor="left", x=0), title='MSE', title_x=0.5)
if data_type.lower()=='panel':
    fig.update_layout(barmode='group', title={"text": ("{} for each model (mean across panels)").format('MSE')})

fig.show(config={'displaylogo': False})

## Creating box plots to compare Model metrics
## ------------------------------------------------------------------------------------------------------------
fig = px.box(error_table, x="model", y="MSE", color="datatype")
display(Markdown("__Box plot__ of error metric : __MSE__"))
fig.update_layout(legend=dict(orientation="h", 
                              yanchor="bottom", y=-0.35, xanchor="left", x=0))
fig.show(config={'displaylogo': False})

## RMSE

## RMSE
## Creating horizontal bar charts to compare Model metrics
## ------------------------------------------------------------------------------------------------------------
track_cell("Model comparison RMSE", flag)

compare_df = pd.DataFrame(error_table.groupby(['model','datatype'],
                                                  group_keys=False).apply(lambda x: x['RMSE'].mean()).reset_index())
compare_df.columns = ['model','datatype','values']
compare_df = compare_df.pivot(index='model', columns='datatype', 
                              values='values').reset_index()
compare_df = compare_df.reset_index(drop = True)
compare_df = compare_df.sort_values(by = 'Validation', ascending = True)
compare_df = compare_df.reset_index(drop = True)

fig = go.Figure(data=[go.Bar(name='Train', y=compare_df['model'], 
                             marker=dict(color='dodgerblue'),
                             x=compare_df['Train'], orientation = 'h'),
                      go.Bar(name='Validation', y=compare_df['model'], 
                             marker=dict(color='darkblue'),
                             x=compare_df['Validation'], orientation = 'h')])

fig.update_layout(legend=dict(orientation="h", yanchor="bottom", 
                              y=-0.35, xanchor="left", x=0), title='RMSE', title_x=0.5)
if data_type.lower()=='panel':
    fig.update_layout(barmode='group', title={"text": ("{} for each model (mean across panels)").format('RMSE')})

fig.show(config={'displaylogo': False})    

## Creating box plots to compare Model metrics
## ------------------------------------------------------------------------------------------------------------
fig = px.box(error_table, x="model", y="RMSE", color="datatype")
display(Markdown("__Box plot__ of error metric : __RMSE__"))
fig.update_layout(legend=dict(orientation="h", 
                              yanchor="bottom", y=-0.35, xanchor="left", x=0))
fig.show(config={'displaylogo': False})

# Forecast (Ex-Ante)

## Data Selection

track_cell("Forecast Read data", flag)
########################################################################################
################################## User Input Needed ###################################
########################################################################################
#Specify data path
data_path = "./Sample_Datasets/Single_Hotel_TS_Test.csv" # options for panel './Sample_Datasets/Hotel_Panel_Test.csv'
filename = data_path.replace("\\","/").split("/")[-1]        
if ".csv" in filename:
    forecast_data = pd.read_csv(data_path)
elif ".xlsx" in filname:
    forecast_data = pd.read_excel(data_path)  ## Defaulted to take sheet 1.Specify sheetnumber with a ',' otherwise

special_chars = r'[?|$|#|@|#|%|!|*|(|)|{|}|^|,|-|.|/|>|<|;|:]'
# lower-case and replace any white-space '_' for every column name
forecast_data.columns = list(map(lambda x:re.sub(special_chars,r'',x.lower().replace(' ','_').replace("'","").replace("|","_").replace('-','_')), forecast_data.columns))

display(Markdown("""<span style="color:blue; font-size: 14px">Below is the summary of the uploaded test data """))
forecast_data.info()

## Data Preprocessing

track_cell("Forecast User Input", flag)
## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
data_type = "non-panel" # options - "panel"/"non-panel"
date_time_freq = 'MS' # options - select from Frequency Selection table based on frequency of data
panel_col = "" # eg - "FAC_ID"
inp_lag = 30 # The ideal value tends to be 0.1% of the input time frame with the maximum value not exceeding 30
target_var = "occupancy"
external_variables = ['max_rooms_capacity', 'avgdailyrate', 'percentgovtnights', 'percentbusiness', 'loyalty_pct'] # options  for panel -['rms_avail_qty', 'age', 'revpar', 'mpi', 'slf']
endog_variables = ['occupancy', 'compet_occupancy'] # options for panel ['occupancy', 'adr']
column_names = forecast_data.columns

if data_type.lower() == 'panel':
    cols_vis = ' || '.join(column_names)
    print(colored("\nColumn Names:",'grey',attrs=['bold']),"\n{}".format(cols_vis))
    panel_shape = []
    for each_panel in panel_ids:
        panel_data = forecast_data.groupby(panel_col).get_group(each_panel)
        panel_shape.append(panel_data.shape[0])

    panel_info = pd.DataFrame()
    panel_info['panel_names'] = panel_ids
    panel_info['size'] = panel_shape

    print_table(panel_info)
else:
    pass
remaining_cols = list(set(column_names) - set(external_variables))
cat_section = []
for i in external_variables:
    if i in cat_cols:
        cat_section.append(i)
if cat_section != []:
    external_variables = [e for e in external_variables if e not in cat_section]
    data_dumm = pd.get_dummies(data[cat_section])
    external_variables.extend(data_dumm.columns)
    data.drop(cat_section, axis=1, inplace=True)
    data = pd.concat([data,data_dumm], axis=1)
    for i in cat_section:
        cat_cols.remove(i)
remaining_cols.remove(target_var)
external_variables_str = " + ".join(external_variables)
formula = target_var + " ~ "+ external_variables_str
clear_output()
display(Markdown("__Selected exogenous variables__: {}".format(external_variables)))
display(Markdown("__Selected endogenous variables__: {}".format(endog_variables)))

clear_output()
display(Markdown("""<span style="color:black;
                    font-size: 15px">The selected datatype is: **{}**""".format(data_type)))
display(Markdown("""<span style="color:black;
                    font-size: 15px">The selected frequency is: **{}**""".format(date_time_freq)))
display(Markdown("""<span style="color:black;
                    font-size: 15px">The selected number of lags is: **{}**""".format(inp_lag)))
display(Markdown("""<span style="color:black;
                    font-size: 15px">The selected panel column is **{}**""".format(panel_col)))
display(Markdown("""<span style="color:black;
                    font-size: 15px">Selected target variable is: **{}**""".format(target_var))) 
display(Markdown("""<span style="color:black;
                    font-size: 15px">Selected exogenous variables (for multivariate forecasting) is/are: **{}**""".format(external_variables))) 
display(Markdown("""<span style="color:black;
                    font-size: 15px">Selected endogenous variables (for multivariate forecasting) is/are: **{}**""".format(endog_variables))) 


## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
date_time_col = "datetime" # options for panel 'yearmonth'
ts_format = "%Y-%m-%d" # options for panel '%Y%m'
is_timestamp = "n" # options - y/n If your dataset has time stamps and you want to retain it along with the date


forecast_data[date_time_col] = pd.to_datetime(forecast_data[date_time_col], format= ts_format)

if '/' in ts_format:
    pass
else:
    if "-" in ts_format:
        ts_format_final = ts_format
    else:
        ts_format_final = '-'.join(ts_format[i:i+2] for i in range(0, len(ts_format), 2))

    forecast_data[date_time_col] = forecast_data[date_time_col].map(lambda x: x.strftime(ts_format_final))

if is_timestamp.lower() == 'y':
    forecast_data[date_time_col] = pd.to_datetime(forecast_data[date_time_col])
elif is_timestamp.lower() == 'n':
    forecast_data[date_time_col] = pd.to_datetime(forecast_data[date_time_col]).dt.date

clear_output()
display(Markdown("The selected date-time column is: __{}__".format(date_time_col)))
display(Markdown('Datetime column converted!'))

forecast_data = forecast_data.sort_values(date_time_col)
display(forecast_data.head().round(4))

if date_time_col in num_cols:
    num_cols.remove(date_time_col)   


if data_type.lower()=='panel':
    display(Markdown('Length of each panels:'))
    print_table(panel_info)
    display(Markdown("""<span style="color:red;
                    font-size: 15px"> Please enter the number of future timestamps to be forecasted. It should be less than the length of your test data across each panels.</span>"""))


## Model Selection

## ------------------------------------------------------------------------------------------------------------
## User Input Required
## ------------------------------------------------------------------------------------------------------------
selection = 'Random Forest' ## Enter Model name

clear_output()
display(Markdown("""<span style="color:blue; font-size: 14px">Selected model</span> : {}""".format(selection)))

if data_type.lower() != 'panel':
    n_periods_ahead = forecast_data.shape[0]
    display(Markdown("""<span style="color:blue; font-size: 14px">Selected number of forecasts required </span>: {}""".format(n_periods_ahead)))

## Prediction

track_cell("Forecast data preparation", flag)
########################################## DO NOT CHANGE THIS #########################################

uni_list = ['Holt-Winters','ARIMA','TBATS','ETS']
multi_list = ['UCM','VAR','ARIMAX']
rem_list = ['Decision Tree','Random Forest','ENET','LASSO','Ridge','GAM','XG Boost','SVR','MARS','PLS']

#######################################################################################################

if data_type.lower()=='panel':
    forecast_panels = selected_panels
    prediction = {}
    if selection in uni_list:
        for pid in tqdm(forecast_panels):
            n_periods_ahead = forecast_data.groupby(panel_col).get_group(pid).shape[0]
            prediction[pid] = dict_models[selection][pid]['Model'].forecast(n_periods_ahead)

    elif selection in multi_list:
        if selection == 'VAR':
            prediction_var = {}
            var_forecast_panels = dict_var.keys()
            for pid in tqdm(var_forecast_panels):
                forecast_data_grp = forecast_data.groupby(panel_col).get_group(pid)
                n_periods_ahead = forecast_data_grp.shape[0]
                prediction_var[pid] = dict_models[selection][pid]['Model'].predict(steps = n_periods_ahead,
                                                               exog = forecast_data_grp[external_variables])
        else:
            if selection == 'ARIMAX':
                for pid in tqdm(forecast_panels):
                    forecast_data_grp = forecast_data.groupby(panel_col).get_group(pid)
                    n_periods_ahead = forecast_data_grp.shape[0]
                    prediction[pid] = dict_models[selection][pid]['Model'].forecast(steps = n_periods_ahead,
                                                                               exog = forecast_data_grp[external_variables])[1] 
            else :       
                for pid in tqdm(forecast_panels):
                    forecast_data_grp = forecast_data.groupby(panel_col).get_group(pid)
                    n_periods_ahead = forecast_data_grp.shape[0]
                    prediction[pid] = dict_models[selection][pid]['Model'].predict(steps = n_periods_ahead,
                                                                               exog = forecast_data_grp[external_variables])            
    elif selection in rem_list:
        for pid in tqdm(forecast_panels):
            forecast_data_grp = forecast_data.groupby(panel_col).get_group(pid)
            prediction[pid] = dict_models[selection][pid]['Model'].predict(forecast_data_grp[external_variables])

    elif selection == 'Prophet':
        for pid in tqdm(forecast_panels):
            future_prophet_data = dict_models[selection][pid]['Model'].make_future_dataframe(periods = n_periods_ahead,
                                                              include_history=True,
                                                              freq=date_time_freq)
            prediction[pid] = dict_models[selection][pid]['Model'].predict(future_prophet_data)  ## Forecasting using prophet

    elif selection == 'QUANTILE':                
        if (not dict_quant):
            print(colored('Model did not execute due to multicolinearity in the data!','red'))
        else:
            for pid in tqdm(forecast_panels):
                forecast_data_grp = forecast_data.groupby(panel_col).get_group(pid)
                prediction[pid] = dict_models[selection][pid]['Model'].predict(forecast_data_grp)
    else:
        for pid in tqdm(forecast_panels):
            forecast_data_grp = forecast_data.groupby(panel_col).get_group(pid)
            prediction[pid] = dict_models[selection][pid]['Model'].predict(forecast_data_grp)


else:
    if selection in uni_list:
        prediction = dict_models[selection]['Model'].forecast(n_periods_ahead)
    elif selection in multi_list: 
        if selection == 'ARIMAX':
            prediction = dict_models[selection]['Model'].forecast(steps = n_periods_ahead,
                                                                 exog = forecast_data[external_variables])
        else:        
            prediction = dict_models[selection]['Model'].predict(steps = n_periods_ahead,
                                                             exog = forecast_data[external_variables])
    elif selection in rem_list:
        prediction = dict_models[selection]['Model'].predict(forecast_data[external_variables])
    elif selection == 'QUANTILE':
        if not dict_quant:
            print(colored('Model did not execute due to multicolinearity in the data!','red'))
        else:
            prediction = dict_models[selection]['Model'].predict(forecast_data)
    elif selection == 'Prophet':
        future_prophet_data = dict_models[selection]['Model'].make_future_dataframe(periods = n_periods_ahead,
                                                                                    include_history=True, 
                                                                                    freq=date_time_freq)
        prediction = dict_models[selection]['Model'].predict(future_prophet_data)
    else:
        prediction = dict_models[selection]['Model'].predict(forecast_data)

clear_output()
display(Markdown('__Processed!__'))

track_cell("Displaying regressor values of the uploaded data",flag) 
if selection not in uni_list:
    display(Markdown("""<span style="color:blue; font-size: 14px">Selected model</span> : {}""".format(selection)))
    show_df = forecast_data[external_variables]
    print_table(show_df.round(4))
else:
    print('\033[35;1m This section is not applicable as univariate method is selected. \033[0m')


# model save
track_cell("Saving predictions and model object",flag)      
if platform.system() == 'Windows':
    predictions_file = '.\\Downloads\\predictions_{}.csv'.format(selection)
    model_file = ".\\Downloads\\{}_modelobj.pickle".format(selection)
else:            
    predictions_file = './Downloads/predictions_{}.csv'.format(selection)            
    model_file = "./Downloads/{}_modelobj.pickle".format(selection)
if data_type.lower() =='panel':            
    if selection == 'VAR':
        predictions_df = pd.Series(data = prediction_var).T
    else:
        predictions_df = pd.Series(data = prediction, index = [0]).T               
    predictions_df.to_csv(predictions_file)                      
    if selection == 'QUANTILE':
        if (not dict_quant):
            clear_output()
            print(colored('Model did not execute due to multicolinearity in the data!','red'))
        else:
            with open(model_file, 'wb') as f:
                pickle.dump(dict_models[selection][pid]['Model'], f)                    
    else :
        with open(model_file, 'wb') as f:
            pickle.dump(dict_models[selection][pid]['Model'], f)                
else:            
    predictions_df = pd.DataFrame(data = prediction).round(4)
    predictions_df[date_time_col] = forecast_data[date_time_col]
    predictions_df.columns = [[target_var,date_time_col]]
    predictions_df = predictions_df.reset_index(drop = True)
    predictions_df.to_csv(predictions_file,index=False)    
    if selection == 'QUANTILE':
        if (not dict_quant):
            clear_output()
            print(colored('Model did not execute due to multicolinearity in the data!','red'))
        else:
            with open(model_file, 'wb') as f:
                pickle.dump(dict_models[selection]['Model'], f)                    
    else :
        with open(model_file, 'wb') as f:
            pickle.dump(dict_models[selection]['Model'], f)
print('\033[35;1m Forecasts and Model object saved in Downloads folder in the current directory \033[0m')

## Goodness of Fit

track_cell("Forecast goodness of fit",flag)

fig = go.Figure()
if data_type.lower() == "panel":
    display(Markdown('__Processing...__'))
    if selection == 'QUANTILE':
        if (not dict_quant):
            print(colored('Model did not execute due to multicolinearity in the data!','red'))
        else:
            for i,id_col in enumerate(tqdm(forecast_panels)):
                visible = True if id_col == list(forecast_panels)[0] else False
                fig = actual_vs_pred_ex_ante(method_name = selection,vis = visible,
                                             train_data = train.groupby(panel_col).get_group(id_col), 
                                             valid_data = valid.groupby(panel_col).get_group(id_col), 
                                             target_var = target_var, 
                                             train_pred = dict_models[selection][id_col]['Fitted'],
                                             valid_pred = dict_models[selection][id_col]['Forecast'],
                                             fc_data = forecast_data.groupby(panel_col).get_group(id_col).reset_index(drop=True),
                                             fc_pred = prediction[id_col], panel = id_col)
            fit_tabs(forecast_panels,selection,True)
    elif selection in ('PLS'):
        for i,id_col in enumerate(tqdm(forecast_panels)):
            visible = True if id_col == list(forecast_panels)[0] else False
            fig = actual_vs_pred_ex_ante(method_name = selection,vis = visible,
                                         train_data = train.groupby(panel_col).get_group(id_col), 
                                         valid_data = valid.groupby(panel_col).get_group(id_col), 
                                         target_var = target_var, 
                                         train_pred = dict_models[selection][id_col]['Fitted'],
                                         valid_pred = dict_models[selection][id_col]['Forecast'],
                                         fc_data = forecast_data.groupby(panel_col).get_group(id_col).reset_index(drop=True),
                                         fc_pred = pd.Series(prediction[id_col].flatten()), 
                                         panel = id_col)
        fit_tabs(forecast_panels,selection,True)
    elif selection == 'VAR':
        forecast_panels_var = dict_var.keys()
        for i,id_col in enumerate(tqdm(forecast_panels_var)):
            visible = True if id_col == list(forecast_panels_var)[0] else False
            fig = actual_vs_pred_ex_ante(method_name = selection,vis = visible,
                                         train_data = train.groupby(panel_col).get_group(id_col), 
                                         valid_data = valid.groupby(panel_col).get_group(id_col), 
                                         target_var = target_var, 
                                         train_pred = dict_models[selection][id_col]['Fitted'][target_var],
                                         valid_pred = dict_models[selection][id_col]['Forecast'][target_var],
                                         fc_data = forecast_data.groupby(panel_col).get_group(id_col).reset_index(drop=True),
                                         fc_pred = prediction_var[id_col][target_var], panel = id_col)
        fit_tabs(forecast_panels_var,selection,True)

    elif selection == 'Prophet':
        forecast_panels_var = dict_var.keys()
        for i,id_col in enumerate(tqdm(forecast_panels_var)):
            visible = True if id_col == list(forecast_panels_var)[0] else False
            fig = actual_vs_pred_ex_ante(method_name = selection,vis = visible,
                                         train_data = train.groupby(panel_col).get_group(id_col), 
                                         valid_data = valid.groupby(panel_col).get_group(id_col), 
                                         target_var = target_var, 
                                         train_pred = dict_models[selection][id_col]['Fitted'],
                                         valid_pred = dict_models[selection][id_col]['Forecast'],
                                         fc_data = forecast_data.groupby(panel_col).get_group(id_col).reset_index(drop=True),
                                         fc_pred = prediction[id_col]['yhat'], panel = id_col)
        fit_tabs(forecast_panels_var,selection,True)

    elif selection == 'ARIMA':
        forecast_panels_var = dict_var.keys()
        for i,id_col in enumerate(tqdm(forecast_panels_var)):
            visible = True if id_col == list(forecast_panels_var)[0] else False
            fig = actual_vs_pred_ex_ante(method_name = selection,vis = visible,
                                         train_data = train.groupby(panel_col).get_group(id_col), 
                                         valid_data = valid.groupby(panel_col).get_group(id_col), 
                                         target_var = target_var, 
                                         train_pred = dict_models[selection][id_col]['Fitted'],
                                         valid_pred = dict_models[selection][id_col]['Forecast'],
                                         fc_data = forecast_data.groupby(panel_col).get_group(id_col).reset_index(drop=True),
                                         fc_pred = prediction[id_col][1], panel = id_col)
        fit_tabs(forecast_panels_var,selection,True)
    else:
        for i,id_col in enumerate(tqdm(forecast_panels)):
            visible = True if id_col == list(forecast_panels)[0] else False
            fig = actual_vs_pred_ex_ante(method_name = selection, vis =visible,
                                         train_data = train.groupby(panel_col).get_group(id_col), 
                                         valid_data = valid.groupby(panel_col).get_group(id_col), 
                                         target_var = target_var, 
                                         train_pred = dict_models[selection][id_col]['Fitted'],
                                         valid_pred = dict_models[selection][id_col]['Forecast'],
                                         fc_data = forecast_data.groupby(panel_col).get_group(id_col).reset_index(drop=True),
                                         fc_pred = prediction[id_col], panel = id_col)
        fit_tabs(forecast_panels,selection,True)
else:
    if selection == 'QUANTILE':
        if not dict_quant:
            print(colored('Model did not execute due to multicolinearity in the data!','red'))
        else:
            fig = actual_vs_pred_ex_ante(method_name= selection, vis=True, 
                                         train_data = train, valid_data = valid, target_var= target_var, 
                                         train_pred = dict_models[selection]['Fitted'],
                                         valid_pred = pd.Series(dict_models[selection]['Forecast']),
                                         fc_data = forecast_data.reset_index(drop=True),
                                         fc_pred = prediction)
    elif selection == 'VAR':
        fig = actual_vs_pred_ex_ante(method_name= selection, vis=True, 
                                     train_data = train, valid_data = valid, target_var= target_var, 
                                     train_pred = dict_models[selection]['Fitted'][target_var],
                                     valid_pred = dict_models[selection]['Forecast'][target_var],
                                     fc_data = forecast_data.reset_index(drop=True),
                                     fc_pred = prediction[target_var])

    elif selection == 'Prophet':
        fig = actual_vs_pred_ex_ante(method_name= selection, vis=True, 
                                     train_data = train, valid_data = valid, target_var= target_var, 
                                     train_pred = dict_models[selection]['Fitted'],
                                     valid_pred = dict_models[selection]['Forecast'],
                                     fc_data = forecast_data.reset_index(drop=True),
                                     fc_pred = prediction['yhat'])

    elif selection == 'ARIMA' or selection == 'ARIMAX':
        fig = actual_vs_pred_ex_ante(method_name= selection, vis=True, 
                                     train_data = train, valid_data = valid, target_var= target_var, 
                                     train_pred = dict_models[selection]['Fitted'],
                                     valid_pred = pd.Series(dict_models[selection]['Forecast']),
                                     fc_data = forecast_data.reset_index(drop=True),
                                     fc_pred = prediction[1])

    elif selection == 'PLS':
        fig = actual_vs_pred_ex_ante(method_name= selection, vis=True,
                                     train_data = train, valid_data = valid, target_var= target_var, 
                                     train_pred = dict_models[selection]['Fitted'],
                                     valid_pred = pd.Series(dict_models[selection]['Forecast']),
                                     fc_data = forecast_data.reset_index(drop=True),
                                     fc_pred = pd.Series(prediction.flatten()))

    else:
        fig = actual_vs_pred_ex_ante(method_name= selection, vis=True, 
                                     train_data = train, valid_data = valid, target_var= target_var, 
                                     train_pred = dict_models[selection]['Fitted'],
                                     valid_pred = dict_models[selection]['Forecast'],
                                     fc_data = forecast_data.reset_index(drop=True),
                                     fc_pred = prediction)

clear_output()
fig.show(config={'displaylogo': False})
