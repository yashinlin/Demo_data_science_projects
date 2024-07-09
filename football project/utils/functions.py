# functions

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import Markdown, display
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xml.etree.ElementTree as ET

def parse_xmlcol_to_df(df, xmlcol):
    """ Converts XML data to dataframe, with tags converted to columns       
    """
    tag_columns = {"match_id": []}  # Initialize with 'id' column

    for i, xml_data in enumerate(df[xmlcol]):
        if xml_data is not None:
            root = ET.fromstring(xml_data)
            tag_columns["match_id"].extend([df["match_id"].iloc[i]] * len(list(root.iter())))
            for elem in root.iter():
                tag_columns.setdefault(elem.tag, []).append(elem.text)

    # Adjust lengths to be the same for each column
    max_length = max(len(v) for v in tag_columns.values())
    for k, v in tag_columns.items():
        if len(v) < max_length:
            tag_columns[k].extend([None] * (max_length - len(v)))

    return tag_columns  # Return the dictionary with extracted tag content

def calculate_hwin_sum(group):
    """ Creates variable hwin_sum, calculated as the sum of wins for a specific home_team, over
    the preceding 35 matches.
    """
    group["hwin_sum"] = (
        group["home_win"].shift().rolling(window=35, min_periods=1).sum()
    )
    group["hwin_sum"] = group["hwin_sum"].fillna(0)
    return group

def calculate_hloss_sum(group):
    """ Creates variable hloss_sum, calculated as the sum of losses for a specific home_team, over
    the preceding 35 matches.
    """
    group["hloss_sum"] = (
        group["away_win"].shift().rolling(window=35, min_periods=1).sum()
    )
    group["hloss_sum"] = group["hloss_sum"].fillna(0)
    return group

def calculate_win_rate(row):
    """ Creates new variable win_rate, calculated as the proportion of wins over the sum of wins and
    losses for a team over the preceding 35 matches it played as home team        
    """
    if (row['hwin_sum'] == 0) and (row['hloss_sum'] == 0):
        return 0
    else:
        return row['hwin_sum'] / (row['hwin_sum'] + row['hloss_sum'])

def round_to_decimal_places(value):
    """Rounds each value in the DataFrame to the desired decimal places
    """
    return round(value, 1)
    
def add_log_transformations(df):
    """Adds log transformations of df predictors and constant for GLM
    """
    for var in df:
        df[f"{var}_log"] = df[var].apply(
            lambda x: x * np.log(x)
        )  # np.log = natural log

def plot_residuals(data, x_var, y_var, title):
    """
    Plots the residuals versus the fitted values in a linear regression model.

    Parameters:
    - data (pandas.DataFrame): The dataset containing the variables.
    - x_var (str): The independent variable in the regression model.
    - y_var (str): The dependent variable in the regression model.
    - title (str): The title for the plot.

    Returns:
    None

    Example:
    >>> plot_residuals(data, 'feature1', 'target', 'Residuals vs Fitted Values')
    """
    
    # Fit the regression model
    model = smf.ols(f"{y_var} ~ {x_var}", data=data).fit()

    # Residuals vs fitted values
    model_fitted_y = model.fittedvalues

    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Plot the residual plot
    plot = sns.residplot(
        x=model_fitted_y,
        y=y_var,
        data=data,
        lowess=False,
        scatter_kws={"alpha": 0.4, "s": 10},
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=ax,
    )

    # Set title and labels
    plot.set_title(title)
    plot.set_xlabel("Fitted values")
    plot.set_ylabel("Residuals")

    # Display the plot
    plt.show()

def fit_linear_regression(data, x, y):
    """
    Fits a linear regression model using the statsmodels library and displays the model summary.

    Parameters:
    - data (pandas.DataFrame): The dataset containing the variables.
    - x (str or list of str): The independent variable(s) for the regression model. If a list is provided,
                              the variables are assumed to be part of a multiple regression.
    - y (str): The dependent variable for the regression model.

    Returns:
    None

    Example:
    >>> fit_linear_regression(data, 'feature1', 'target')
    >>> fit_linear_regression(data, ['feature1', 'feature2'], 'target')
    """
    # Create formula string
    if isinstance(x, list):
        formula = f"{y} ~ {' + '.join(x)}"
    else:
        formula = f"{y} ~ {x}"

    # Fit linear regression model
    model = smf.ols(formula=formula, data=data).fit()

    # View model summary
    display(
        Markdown(
            f"\n\n**Linear Regression: {y} ~ {x}**  ----------------------------------------"
        )
    )
    display(model.summary())

    # Residual mean squared error
    rmse_resid = np.sqrt(model.mse_resid)

    # Calculate and display the Residual Mean Standard Error
    display(
        Markdown(
            f"\n\n**RMSE of residuals: {rmse_resid}** ----------------------------------------\n\n"
        )
    )

def calculate_goal_diff(group):
    """
    Calculates the cumulative sum of goal differences over a rolling window of the last 35 matches.

    Parameters:
    - group (pandas.DataFrame): A group of data containing information about goal differences.

    Returns:
    pandas.DataFrame: The input group DataFrame with an additional column 'hx_goal_diff' representing
                     the cumulative sum of goal differences over a rolling window.

    Example:
    >>> data_group = calculate_goal_diff(data_group)
    """
    group["hx_goal_diff"] = (
        group["goal_diff"].shift().rolling(window=35, min_periods=1).sum()
    )
    return group

def plot_pairplot(data, target, predictors):
    """
    Creates and displays a pair plot with regression lines for the specified target variable
    and a selection of predictor variables.

    Parameters:
    - data (pandas.DataFrame): The dataset containing the variables.
    - target (str): The target variable for the pair plot.
    - predictors (list of str): The list of predictor variables to be included in the pair plot.

    Returns:
    None

    Example:
    >>> plot_pairplot(data, 'target_variable', ['predictor1', 'predictor2', 'predictor3'])
    """
    sns.set(font_scale=3)
    plot_kws = {"line_kws": {"color": "red"}}  # Pair plots
    sns.set(style="ticks", color_codes=True)

    # Combine the target variable with the selected predictors
    selected_variables = [target] + predictors
    selected_data = data[selected_variables]

    # Create a pair plot
    pair_plot = sns.pairplot(
        selected_data,
        kind="reg",
        diag_kind="kde",
        plot_kws={
            "line_kws": {"color": "red", "linewidth": 1},
            "scatter_kws": {"alpha": 0.08, "s": 2},
        },
        height=4,
        aspect=0.6,
    )

    pair_plot.fig.set_size_inches(14, 10)
    plt.show()

def looker_link(url, text):
    return f'<a href="{url}" target="_blank">{text}</a>'

print("End of functions module")