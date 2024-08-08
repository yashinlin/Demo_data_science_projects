import jupyter_black
from datetime import datetime

from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.encoding import CountFrequencyEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

# Notebook 01

def process_and_show_report(filename):
    """
    Reads a CSV file from the 'data' directory, creates a report, and shows it in the browser.

    Parameters:
    - filename: str, name of the CSV file to process.
    """
    base_dir = "data"
    file_path = os.path.join(base_dir, filename)
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}, Columns: {df.columns}")
    create_report(df).show_browser()

def find_duplicates(file_path, chunksize=10000, index_col=[0]):
    """
    Finds duplicate rows in a DataFrame file, handling large files efficiently using chunking.

    Args:
       file_path (str): Path to the DataFrame file to be processed.
       chunksize (int, optional): Number of rows to read in each chunk. Defaults to 10000.

    Returns:
       pd.DataFrame: A DataFrame containing the duplicate rows found in the file.
    """

    duplicate_rows = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        duplicates_in_chunk = chunk[chunk.duplicated()]
        duplicate_rows.append(duplicates_in_chunk)

    all_duplicates = pd.concat(duplicate_rows)
    return all_duplicates

def rename_columns(df):
    """
    Rename columns of a DataFrame according to the provided mapping and convert all column names to lowercase.

    Parameters:
    - df: pandas.DataFrame, the original DataFrame whose columns need renaming.

    Returns:
    - pandas.DataFrame: A new DataFrame with renamed and lowercase columns.
    """
    # Define the column name mapping
    column_mapping = {
        "FLAG_WORK_PHONE": "work_phone",
        "HOUR_APPR_PROCESS_START": "time_applied",
        "EXT_SOURCE_3": "score_3",
        "EXT_SOURCE_2": "score_2",
        "EXT_SOURCE_1": "score_1",
        "DAYS_REGISTRATION": "registration_age_d",
        "REGION_RATING_CLIENT": "reg_rating",
        "REGION_RATING_CLIENT_W_CITY": "reg_city_rating",
        "FLAG_OWN_REALTY": "realty",
        "REGION_POPULATION_RELATIVE": "pop_density",
        "ORGANIZATION_TYPE": "org",
        "OCCUPATION_TYPE": "occupation",
        "CNT_CHILDREN": "kids",
        "DAYS_EMPLOYED": "job_age_d",
        "NAME_INCOME_TYPE": "income_type",
        "AMT_INCOME_TOTAL": "income",
        "DAYS_ID_PUBLISH": "id_age_d",
        "NAME_HOUSING_TYPE": "housing",
        "FLAG_PHONE": "home_phone",
        "AMT_GOODS_PRICE": "goods",
        "CODE_GENDER": "gender",
        "NAME_FAMILY_STATUS": "family_status",
        "CNT_FAM_MEMBERS": "family_num",
        "FLAG_EMP_PHONE": "emp_phone",
        "FLAG_EMAIL": "email",
        "NAME_EDUCATION_TYPE": "education",
        "WEEKDAY_APPR_PROCESS_START": "day_applied",
        "AMT_CREDIT": "credit_loan",
        "AMT_REQ_CREDIT_BUREAU_YEAR": "credit_inq_yr",
        "AMT_REQ_CREDIT_BUREAU_WEEK": "credit_inq_wk",
        "AMT_REQ_CREDIT_BUREAU_QRT": "credit_inq_qtr",
        "AMT_REQ_CREDIT_BUREAU_MON": "credit_inq_mth",
        "AMT_REQ_CREDIT_BUREAU_HOUR": "credit_inq_hr",
        "AMT_REQ_CREDIT_BUREAU_DAY": "credit_inq_day",
        "NAME_CONTRACT_TYPE": "contract_type",
        "FLAG_CONT_MOBILE": "cell_reachable",
        "FLAG_MOBIL": "cell_phone",
        "OWN_CAR_AGE": "car_age",
        "FLAG_OWN_CAR": "car",
        "APARTMENTS_MODE": "apt_mode",
        "APARTMENTS_AVG": "apt",
        "AMT_ANNUITY": "annuity",
        "DAYS_BIRTH": "age_d",
        "NAME_TYPE_SUITE": "accompanied",
        "FLAG_DOCUMENT_2": "doc2",
        "FLAG_DOCUMENT_3": "doc3",
        "FLAG_DOCUMENT_4": "doc4",
        "FLAG_DOCUMENT_5": "doc5",
        "FLAG_DOCUMENT_6": "doc6",
        "FLAG_DOCUMENT_7": "doc7",
        "FLAG_DOCUMENT_8": "doc8",
        "FLAG_DOCUMENT_9": "doc9",
        "FLAG_DOCUMENT_10": "doc10",
        "FLAG_DOCUMENT_11": "doc11",
        "FLAG_DOCUMENT_12": "doc12",
        "FLAG_DOCUMENT_13": "doc13",
        "FLAG_DOCUMENT_14": "doc14",
        "FLAG_DOCUMENT_15": "doc15",
        "FLAG_DOCUMENT_16": "doc16",
        "FLAG_DOCUMENT_17": "doc17",
        "FLAG_DOCUMENT_18": "doc18",
        "FLAG_DOCUMENT_19": "doc19",
        "FLAG_DOCUMENT_20": "doc20",
        "FLAG_DOCUMENT_21": "doc21"
    }

    renamed_df = df.copy()
    renamed_df = renamed_df.rename(columns=column_mapping)
    renamed_df.columns = renamed_df.columns.str.lower()    
    return renamed_df

def downsize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all float columns in a pandas DataFrame to float16 data type.

    Args:
      df (pandas.DataFrame): The DataFrame to convert.

    Returns:
      pandas.DataFrame: The DataFrame with all float columns converted to float16.
    """
    for col in df.columns:
        if df[col].dtype == np.float64 or df[col].dtype == np.int64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == object:
            df[col] = df[col].astype("category")
    return df

def take_sample(df: pd.DataFrame, target: str, test_size: float) -> pd.DataFrame:   
    X = df.drop(columns=target, axis=1)
    y = df[target]
    
    X_big, X_small, y_big, y_small = train_test_split(
        X, y, test_size=test_size, random_state=40, stratify=y
    )

    return X_small.merge(y_small, left_index=True, right_index=True)

def select_cols(df: pd.DataFrame, cols: list)-> pd.DataFrame:
    """
    Selects a subset of columns from a DataFrame.

    Args:
        data: A pandas DataFrame.
        cols: A list of column names to keep.

    Returns:
        A new DataFrame containing only the specified columns.
    """
    if not isinstance(cols, list):
        cols = list(cols)
        
    df_reduced = df[cols]   
    return df_reduced

def define_data_structures(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Defines data structures for exploratory data analysis (EDA).

    Parameters:
        df (pd.DataFrame): The input DataFrame containing features and target variable.
        target (str): The name of the target variable column.

    Returns:
        tuple: A tuple containing the following data structures:
            - X_train (pd.DataFrame): Feature matrix (excluding the target variable).
            - y_train (pd.Series): Target variable.
            - X_train_num (pd.DataFrame): Subset of features with numeric data types.
            - X_train_num_lst (list): List of column names for numeric features.
            - X_train_cat (pd.DataFrame): Subset of features with object or category data types.
            - X_train_cat_lst (list): List of column names for categorical features.
            - train (pd.DataFrame): Original DataFrame (including both features and target variable).
    """
    train = df
    X_train = train.drop(columns=[target], axis=1)
    y_train = train[[target]]
    
    X_train_num = X_train.select_dtypes("number")
    X_train_num_lst = list(X_train_num)

    X_train_cat = X_train.select_dtypes(["object", "category"])
    X_train_cat_lst = list(X_train_cat)

    return (
        X_train,
        y_train,
        X_train_num,
        X_train_num_lst,
        X_train_cat,
        X_train_cat_lst,
        train,
    )

def process_dataset_eda(dataset: pd.DataFrame, target: str, test_size: float, selected_features: list):
    """
    Processes a dataset by downscaling dtypes, taking a sample, selecting columns

    Parameters:
    - dataset: The dataset to process.
    - target: The target column for sampling.
    - test_size: The proportion of the dataset to reserve for testing.
    - selected_features: Columns to select.

    Returns:
    - A tuple containing the processed dataset, features, labels, numerical features, categorical features,
      and lists of numerical and categorical features.
    """
    return (
        dataset.pipe(downsize_dtypes)
        .pipe(take_sample, target=target, test_size=test_size)
        .pipe(select_cols, cols=selected_features)
        .pipe(define_data_structures, target=target)
    )

def create_feature_ratios(df):
    """
    Create new feature ratios and columns based on existing features in the dataframe.
    
    Args:
    df (pd.DataFrame): Input dataframe containing the required columns.
    
    Returns:
    pd.DataFrame: Dataframe with newly created features.
    """
    df = df.copy()
    df["credit_utilization_ratio"] = df["credit_loan"] / df["income"]
    df["credit2income"] = df["goods"] / df["income"]
    df["annuity_to_income_ratio"] = df["annuity"] / df["income"]
    df["days_since_last_id_change"] = df["job_age_d"] - df["id_age_d"]
    df["family_size"] = df["family_num"] + df["kids"]    
    df["phone_accessibility"] = df["cell_reachable"] + df["home_phone"] + df["cell_phone"]
    df["recent_credit_inquiries"] = df[["credit_inq_day", "credit_inq_hr"]].sum(axis=1)
    return df

def plot_lineplots(data, columns):
    """
    Plots line plots for each column against the target variable.
    Parameters:
    - data: DataFrame containing the training data.
    - columns: List of column names to be plotted against the target variable.
    """
    plt.figure(figsize=(12, 6))

    for column in columns:
        sns.lineplot(data=data, x=column, y="target", label=column)

    plt.title(
        "Figure 4. Proportion of approved loans by number of credit inquiries in the timeframe indicated"
    )
    plt.xlabel(
        "Number of credit inquiries in the time period immediately preceding loan application"
    )
    plt.ylabel("Proportion of approved loans")
    
def correlation_bar(df: pd.DataFrame, target: str, title: str, figsize=(1, 22)) -> None:
    """
    Creates a heatmap visualization of correlations between a specified feature and other features in a DataFrame.

    This function generates a heatmap using Seaborn library to visualize the correlation coefficients between a target feature and 
    all other features in a DataFrame. It allows for customization of the correlation calculation method (`kind`), figure size (`figsize`), and plot title (`title`).

    Args:
    df (pd.DataFrame): The input DataFrame containing the features for correlation analysis.
    feature (str): The name of the feature to calculate correlations with.
    title (str): The title to display for the generated heatmap.
    kind (str, default="kendall"): The correlation coefficient calculation method. Valid options include 'spearman', 'pearson', or 'kendall' (default).
    figsize (tuple, default=(1, 16)): A tuple of two integers representing the width and height of the generated figure in inches.

    Returns:
    None: This function creates a visualization (heatmap) and does not return any value.
    """
    df_phik_bar = df[[target]].sort_values(by=target, ascending=False).drop(target)
    fig, ax = plt.subplots(figsize=(1, 22))
    heatmap = sns.heatmap(
        df_phik_bar,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap=sns.color_palette("coolwarm"),
        cbar=True,
        annot_kws={"size": 7},
        ax=ax,
    )
    heatmap.set_title(title)
    ax.tick_params(axis="y", labelsize=5)
    return df_phik_bar


def split_val_test(
    X: pd.DataFrame, y: pd.DataFrame, train_size, val_size, test_size
) -> pd.DataFrame:
    """
    Splits the dataset into training, validation and testing sets randomly.

    Parameters:
    - X: Feature DataFrame
    - y: Target Series or DataFrame
    - test_size: float, test set size
    - date: str, date to define earliest dataset date for prediction

    Returns:
    - X_train, X_test, y_train, y_test
    """
    assert (
        train_size + val_size + test_size == 1
    ), "Train, validation, and test sizes must add up to 1"

    X_train_0, X_test, y_train_0, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_0, y_train_0, test_size=val_ratio, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Notebook 02 ====================================================================================

def downsample_majority_class(df: pd.DataFrame, target: str, downsample_mult_factor: int):
    """
    Downsamples the majority class in a given DataFrame to balance the dataset based on the presence of a 
    specific target value and then downsamples this class by a specified multiplier factor. The remaining 
    instances of the majority class are preserved. 
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset. It is expected to have a column named
                         'target' that specifies the class membership of each instance.
    - target (str): The name of the column in the DataFrame that indicates the class membership. Typically,
                    this column contains binary values (e.g., 0 and 1) where one value represents the majority
                    class and the other represents the minority class.
    - downsample_mult_factor (int): The multiplier factor by which the majority class will be reduced. This factor
                                   determines the reduction ratio of the majority class relative to the minority
                                   class. For example, a factor of 2 means that the number of instances in the
                                   majority class will be halved.

    Returns:
    - df_downsampled (pd.DataFrame): A new DataFrame that is a result of downsampling the majority class. This
                                     DataFrame contains both the instances of the majority class (preserved) and
                                     a subset of the instances of the minority class (downsampled). The total
                                     number of instances in the returned DataFrame remains unchanged from the
                                     original DataFrame.

    Note:
    - The resampling process is performed without replacement, meaning that each instance in the minority
      class is unique and not duplicated.
    - The random state is fixed to ensure reproducibility of the downsampling process.
    """
    mask = df[target] == 1  
    target_1 = df[mask]
    target_0 = df[~mask]
    
    # Check if the desired number of rows for the majority class exceeds the available rows
    n_maj_rows = len(target_1) * downsample_mult_factor
    if n_maj_rows > len(target_0):
        raise ValueError("Reduce the desired multiplicative factor of majority class (arguments: df, target (str), *factor*)")
    
    target_0_downsampled = resample(
        target_0,
        replace=False,
        n_samples=n_maj_rows,
        random_state=42,
    )

    df_downsampled = pd.concat([target_1, target_0_downsampled])
    return df_downsampled

def process_dataset(dataset: pd.DataFrame, target: str, downsample_mult_factor:float, sample_size: float, selected_features: list):
    """
    Processes a dataset by downscaling dtypes, taking a sample, selecting columns

    Parameters:
    - dataset: The dataset to process.
    - target: The target column for sampling.
    - test_size: The proportion of the dataset to reserve for testing.
    - selected_features: Columns to select.

    Returns:
    - A tuple containing the processed dataset, features, labels, numerical features, categorical features,
      and lists of numerical and categorical features.
    """
    return (
        dataset.pipe(rename_columns)
        .pipe(downsize_dtypes)
        .pipe(take_sample, target=target, test_size=sample_size)
        .pipe(downsample_majority_class, target, downsample_mult_factor)
        .pipe(select_cols, cols=selected_features)
        .pipe(define_data_structures, target=target)
    )

class ClampOutliersTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for clamping outliers in a DataFrame.

    This transformer identifies and clamps outliers in a DataFrame (X) using the
    Interquartile Range (IQR) method. It calculates the quartiles (Q1 and Q3) and
    the IQR for each feature. Outliers are then defined as values that fall outside
    a specified factor (default 1.5) times the IQR away from the quartiles. The
    transformer clamps these outliers to the nearest non-outlier values (either
    Q1 - factor * IQR or Q3 + factor * IQR).

    Inherits from scikit-learn's `BaseEstimator` and `TransformerMixin` classes for
    compatibility with pipelines.
    """

    def __init__(self, factor=1.5):
        """
        Initializes the transformer with a user-defined factor for outlier clamping.

        Args:
            factor (float, optional): The factor to multiply the IQR when defining
                outlier thresholds. Defaults to 1.5 (values outside 1.5 * IQR are clamped).
        """
        self.factor = factor

    def fit(self, X=pd.DataFrame, y=None):
        """
        Learns the quartiles and IQR for each feature in the input data.

        Args:
            X (pd.DataFrame): The input DataFrame containing features.
            y (pd.Series, optional): Ignored.

        Returns:
            self: The fitted transformer object.

        Calculates the 1st quartile (Q1) and 3rd quartile (Q3) for each feature in the
        input DataFrame (X). Also, computes the IQR (Q3 - Q1). These values are stored
        as attributes (`q1_`, `q3_`, and `iqr_`) for later use in the `transform` method.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.q1_ = X.quantile(0.25)
        self.q3_ = X.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_
        return self

    def transform(self, X):
        """
        Transforms (clamps) outliers in the input data based on the learned quartiles and IQR.
    
        Args:
            X (pd.DataFrame): The input DataFrame containing features.
    
        Returns:
            pd.DataFrame: A new DataFrame where outlier values in each feature have
                been clamped to the nearest non-outlier value (based on Q1, Q3, IQR, and factor).
    
        Calculates the lower and upper bounds for outlier clamping using the learned
        quartiles, IQR, and the user-defined factor. Applies clipping (replacing outliers)
        to the input DataFrame (X) along the columns (axis=1). This can help reduce the
        influence of extreme values on downstream machine learning tasks.
        """
        min_val = self.q1_ - (self.iqr_ * self.factor)
        max_val = self.q3_ + (self.iqr_ * self.factor)

        if isinstance(X, pd.DataFrame):
            X_clamped = X.apply(lambda col: col.clip(lower=min_val[col.name], upper=max_val[col.name]), axis=0, result_type='broadcast')
        elif isinstance(X, np.ndarray):
            X_clamped = np.clip(X, a_min=min_val.values, a_max=max_val.values)
        else:
            raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")
    
        return X_clamped
    
def create_preprocessing_pipeline(num_columns, cat_columns, y_train):
    num_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clamp_outliers", ClampOutliersTransformer()),
            ("ss", StandardScaler()),
        ]
    )

    cat_transform = Pipeline(
        steps=[
            (
                "replace_missing",
                SimpleImputer(strategy="constant", fill_value="missing_value"),
            ),
            ("cf_encoder", CountFrequencyEncoder(encoding_method="frequency")),

        ]
    )

    pp_transformer = ColumnTransformer(
        [("num", num_transform, num_columns), ("cat", cat_transform, cat_columns)
        ],
        verbose_feature_names_out=False,
        remainder="drop",
    )

    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=np.ravel(y_train)
    )
    class_weights = dict(zip(np.unique(y_train), weights))

    return pp_transformer, class_weights

def get_high_importance_features(cb_estimator, X_train):
    """
    Extracts high importance features based on the change in prediction values.

    Parameters:
    - cb_estimator: An instance of a classifier or regressor with a method `get_feature_importance`.
    - X_train: A DataFrame containing the training data.

    Returns:
    - A list of high importance feature names.
    """
    importances = cb_estimator.get_feature_importance(type="PredictionValuesChange")
    feature_importances = pd.Series(importances, index=X_train.columns).sort_values(
        ascending=False
    )
    features_mask = feature_importances > 0
    feature_imp_df = pd.DataFrame(
        feature_importances[features_mask], columns=["Importances"]
    )
    feature_imp_hi_lst = feature_imp_df.index.tolist()
    return feature_imp_df, feature_imp_hi_lst

# Notebook 03 =====================================================================


print("End of utils module")
