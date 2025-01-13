import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Union, List
from loguru import logger
import gc
from scipy.stats import yeojohnson
import numpy as np



# ==================================================================================================================== #
#                                                        GENERAL                                                       #
# ==================================================================================================================== #
def plot_dtype_distribution(df: pd.DataFrame, main_title: str = "Data Type Distribution") -> None:
    """
    Generates a pie chart visualizing the distribution of data types in the given dataframe
    and a bar chart showing the count of each data type with consistent colors.

    Args:
        df (pd.DataFrame): The input dataframe for which the data type distribution is plotted.
        main_title (str): The main title to be displayed above the combined plots. Defaults to "Data Type Distribution".

    Returns:
        None: This function displays the pie and bar charts and does not return any value.

    Raises:
        ValueError: If the dataframe is empty or has no columns.
    """
    # Check if dataframe is empty or has no columns
    if df.empty or df.shape[1] == 0:
        logger.error("The dataframe is either empty or has no columns.")
        raise ValueError("The dataframe must not be empty and should contain columns.")

    # Get data type counts
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ['dtype', 'count']

    # Define a consistent color palette
    unique_dtypes = dtype_counts['dtype']
    palette = sns.color_palette("colorblind", len(unique_dtypes))
    color_mapping = {dtype: palette[i] for i, dtype in enumerate(unique_dtypes)}

    # Set title properties
    title = {"family": "Arial", "color": "black", "weight": "bold", "size": 18}

    # Configure seaborn style
    sns.set_style("whitegrid")

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot pie chart
    dtype_counts.set_index('dtype')['count'].plot(
        kind="pie",
        autopct="%1.1f%%",
        colors=[color_mapping[dtype] for dtype in dtype_counts['dtype']],
        wedgeprops=dict(width=0.6, edgecolor='w'),
        textprops=dict(color='black', size=12, weight='bold'),
        startangle=90,
        pctdistance=0.7,
        ax=axes[0]
        )
    axes[0].set_title("Distribution of Data Types", fontdict=title)
    axes[0].legend().remove()

    # Plot bar chart
    sns.barplot(
        x='dtype',
        y='count',
        data=dtype_counts,
        ax=axes[1],
        palette=[color_mapping[dtype] for dtype in dtype_counts['dtype']]
        )
    axes[1].set_title("Count of Each Data Type", fontdict=title)
    axes[1].set_xlabel("Data Type")
    axes[1].set_ylabel("Count")

    # Annotate bar plot
    for p in axes[1].patches:
        axes[1].annotate(
            f'{int(p.get_height()):,}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='baseline',
            fontsize=12,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
            )

    # Add the main title
    fig.suptitle(main_title, fontsize=20, fontweight='bold', color='black', family="Arial")

    # Adjust layout and show plots
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust layout to fit the main title
    plt.show()

    logger.debug("Pie and bar charts displayed successfully.")

# Example usage:
# plot_dtype_distribution(df)


# ==================================================================================================================== #
#                                                     DISTRIBUTION                                                     #
# ==================================================================================================================== #
def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Plots the distribution of the TARGET variable as a pie chart and a bar chart,
    with consistent colors, annotations, and a legend for the pie chart.

    Args:
        df (pd.DataFrame): The dataframe containing the TARGET column.

    Raises:
        ValueError: If the dataframe is empty or does not contain the TARGET column.
    """
    # Validate input dataframe
    if df.empty or 'TARGET' not in df.columns:
        logger.error("The dataframe is either empty or missing the 'TARGET' column.")
        raise ValueError("The dataframe must contain a non-empty 'TARGET' column.")

    # Calculate the percentage and counts of each value in TARGET
    target_counts = df['TARGET'].value_counts()
    target_percentages = target_counts / len(df) * 100

    # Define the figure with a 1-row, 2-column layout
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Plot 1: Pie chart for TARGET distribution
    wedges, texts, autotexts = axes[0].pie(
        target_percentages,
        labels=target_percentages.index,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        startangle=90,
        colors=sns.color_palette('colorblind'),
        textprops={'fontsize': 12, 'weight': 'bold'}
        )
    axes[0].set_title("Distribution", fontsize=14, weight='bold')
    axes[0].axis('equal')  # Draw the pie as a circle

    # Add a legend for the pie chart
    axes[0].legend(
        wedges,
        [f"{idx} ({count:,})" for idx, count in zip(target_counts.index, target_counts)],
        title="TARGET",
        loc="upper right",
        fontsize=10
        )

    # Plot 2: Bar plot for TARGET counts
    sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        ax=axes[1],
        palette="colorblind"
        )
    axes[1].set_title("Count", fontsize=14, weight='bold')
    axes[1].set_xlabel("TARGET", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)

    # Annotate bar plot with counts
    for p, count in zip(axes[1].patches, target_counts.values):
        axes[1].annotate(
            f'{count:,}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='baseline',
            fontsize=11,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
            )

    # Overall title for the figure
    fig.suptitle('Distribution of TARGET Variable', fontsize=16, fontweight='bold', color='black', family="Arial")

    # Adjust layout and display the plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Log success message
    logger.debug("TARGET distribution plots displayed successfully.")


def plot_variable_and_target_distribution(df: pd.DataFrame, target_col: str = 'TARGET', category_col: str = None, title: str = 'Insert_title') -> None:
    """
    Plots the distribution of a specified categorical variable by target variable
    and its overall distribution with count annotations.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        target_col (str): The name of the target column (default is 'TARGET').
        category_col (str): The name of the categorical column to plot (default is None, must be provided).
        title (str): The title to be used in the plot (default is 'Insert_title').

    Returns:
        None: Displays the plots.

    Raises:
        ValueError: If the dataframe is missing necessary columns, is empty, or if category_col is not provided.
    """
    # Validate input parameters
    if category_col is None:
        logger.error("Category column must be specified.")
        raise ValueError("category_col cannot be None.")

    if df.empty:
        logger.error("The dataframe is empty.")
        raise ValueError("The dataframe must not be empty.")

    required_columns = {target_col, category_col}
    if not required_columns.issubset(df.columns):
        logger.error(f"The dataframe must contain the columns: {required_columns}")
        raise ValueError(f"Dataframe must contain columns: {required_columns}")

    # Define title properties
    title_props = {"family": "Arial", "color": "black", "weight": "bold", "size": 16}

    # Get unique categories, ignoring NaN values
    unique_categories = df[category_col].dropna().unique()

    # Define color palette
    palette = sns.color_palette("colorblind", len(unique_categories))
    color_mapping = {cat: palette[i] for i, cat in enumerate(unique_categories)}
    category_order = sorted(unique_categories)

    # Set up the figure
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    # Plot 1: Distribution of category_col by target_col
    sns.countplot(
        data=df,
        x=target_col,
        hue=category_col,
        hue_order=category_order,
        ax=axs[0],
        palette=color_mapping
        )
    axs[0].set_title('Distribution by TARGET', fontdict=title_props)
    axs[0].set_xlabel(target_col, fontsize=12)
    axs[0].set_ylabel(f'Count of {category_col}', fontsize=12)
    axs[0].legend(title=category_col, fontsize=10, title_fontsize=12)

    # Plot 2: Overall distribution of category_col
    sns.countplot(
        data=df,
        x=category_col,
        order=category_order,
        ax=axs[1],
        palette=color_mapping
        )
    axs[1].set_title(f'Distribution of {category_col}', fontdict=title_props)
    axs[1].set_xlabel(category_col, fontsize=12)
    axs[1].set_ylabel('Count', fontsize=12)

    # Rotate x-ticks if any category name is too long
    if any(len(str(cat)) > 3 for cat in category_order):
        axs[1].tick_params(axis='x', rotation=45)

    # Annotate bar counts
    for p in axs[1].patches:
        axs[1].annotate(
            f'{int(p.get_height()):,}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='baseline',
            fontsize=11,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
            )

    # Add a main title using the `title` parameter
    fig.suptitle(title, fontsize=20, fontweight='bold', color='black', family="Arial")

    # Adjust layout and show plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    logger.info(f"Distribution plots for {category_col} by {target_col} generated successfully.")

# Example usage:
# plot_distribution(df= df, target_col = 'TARGET', category_col = 'NAME_CONTRACT_TYPE', title= 'Loan Type')


from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

@logger.catch
def plot_numerical_distribution_boxplot(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Plots histograms and boxplots for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (List[str]): A list of column names for which to plot histograms and boxplots.

    Returns:
        None: Displays the plots.
    """
    # Filter for valid numeric columns
    valid_cols: List[str] = [
        col for col in columns
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not valid_cols:
        logger.warning("No valid numeric columns found. Exiting the plotting function.")
        return

    logger.info(f"Plotting for columns: {valid_cols}")

    num_columns: int = 4  # Each row will have 2 histograms and 2 boxplots
    num_rows: int = len(valid_cols)

    plt.figure(figsize=(20, 5 * num_rows))

    for i, col in enumerate(valid_cols):
        # Drop NaN values
        data = df[col].dropna()

        if data.empty:
            logger.warning(f"No data available for column {col}. Skipping...")
            continue

        # Plot histogram
        plt.subplot(num_rows, num_columns, 2 * i + 1)
        sns.histplot(data, kde=False, color='gray', binwidth=(data.max() - data.min()) / 30)
        plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(data.median(), color='blue', linestyle='dashed', linewidth=1.5, label='Median')
        plt.title(f'Histogram of {col}')
        plt.legend()

        # Plot boxplot
        plt.subplot(num_rows, num_columns, 2 * i + 2)
        sns.boxplot(x=data)
        plt.title(f'Boxplot of {col}')

    plt.tight_layout(pad=3.0)
    plt.show()



# == UNUSED ============================================================================================================

def plot_distribution_and_boxplot(data: pd.DataFrame, columns: Union[str, List[str]]) -> None:
    """Plots the distribution and box plot for specified columns in a DataFrame, and logs the whisker values and counts of outliers.

    This function creates a histogram and a boxplot for each column specified. It also calculates the lower and upper whiskers
    based on the interquartile range (IQR) and logs these values along with the counts of outliers below and above these whiskers.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        columns (Union[str, List[str]]): A single column name or a list of column names to plot.

    Returns:
        None: This function does not return any value. It plots figures and logs information about the plots.
    """
    if isinstance(columns, str):
        columns = [columns]  # Convert to list if only one column name is provided

    valid_cols = [col for col in columns if col in data.columns and data[col].dtype in ['float64', 'int64']]

    if not valid_cols:
        logger.warning("No valid columns provided or columns do not exist in the DataFrame.")
        return

    num_rows = len(valid_cols)

    plt.figure(figsize=(20, 5 * num_rows))

    for i, col in enumerate(valid_cols):
        data_col = data[col].dropna()

        q1 = data_col.quantile(0.25)
        q3 = data_col.quantile(0.75)
        iqr = q3 - q1
        lower_whisker = max(q1 - 1.5 * iqr, data_col.min())  # Ensure the lower whisker is not less than the minimum value
        upper_whisker = min(q3 + 1.5 * iqr, data_col.max())  # Ensure the upper whisker does not exceed the maximum value

        outliers_below = (data_col < lower_whisker).sum()
        outliers_above = (data_col > upper_whisker).sum()

        plt.subplot(num_rows, 2, 2 * i + 1)
        sns.histplot(data_col, kde=False, color='gray', binwidth=(data_col.max() - data_col.min()) / 30)
        plt.axvline(data_col.mean(), color='red', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(data_col.median(), color='blue', linestyle='dashed', linewidth=1.5, label='Median')
        plt.title(f'Histogram of {col}')
        plt.legend()

        plt.subplot(num_rows, 2, 2 * i + 2)
        sns.boxplot(x=data_col)
        plt.title(f'Boxplot of {col}')

        logger.info(f"Completed plotting distributions for {col}.")
        logger.info(f"{col}: Lower Whisker = {lower_whisker}, Outliers below = {outliers_below}")
        logger.info(f"{col}: Upper Whisker = {upper_whisker}, Outliers above = {outliers_above}")

    plt.tight_layout(pad=3.0)
    plt.show()

# Example usage:
# plot_distribution_boxplot(df, 'AMT_ANNUITY')
# plot_distribution_boxplot(df, ['AMT_CREDIT', 'AMT_ANNUITY'])


def plot_transformations(data: pd.DataFrame, column: str):
    """
    Applies suitable transformations to a specified column in the DataFrame based on its values,
    plots histograms and boxplots for each, and reports the number of outliers before and after each transformation.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to apply transformations on.

    Returns:
        None: Plots histograms and boxplots for each transformed version of the column, logs outlier counts.
    """
    if column not in data.columns or data[column].dtype not in ['float64', 'int64']:
        logger.warning(f"Column {column} is not valid or does not exist in the DataFrame.")
        return

    # Clean data from NaNs and Infs
    clean_data = data[column].replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure checks for transformations are accurate
    positive_values = clean_data[clean_data >= 0]
    non_zero_values = clean_data[clean_data != 0]

    # Determine outliers in the original data
    q1 = clean_data.quantile(0.25)
    q3 = clean_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(clean_data.min(), q1 - 1.5 * iqr)
    upper_bound = min(clean_data.max(), q3 + 1.5 * iqr)
    original_outliers = ((clean_data < lower_bound) | (clean_data > upper_bound)).sum()
    logger.info(f"Original data for {column} contains {original_outliers} outliers.")

    # Preparing transformations
    transformations = {}
    if positive_values.size > 0:  # Check if there are positive values for log and sqrt
        transformations['log'] = np.log(positive_values + 1)
        transformations['sqrt'] = np.sqrt(positive_values)
    if non_zero_values.size > 0:  # Check if there are non-zero values for inverse
        transformations['inverse'] = 1 / non_zero_values
    # Applying Yeo-Johnson which works with any real numbers
    transformations['yeo-johnson'] = pd.Series(yeojohnson(clean_data)[0])

    plt.figure(figsize=(15, len(transformations) * 5))

    for i, (key, transformed_col) in enumerate(transformations.items(), 1):
        # Determine outliers in the transformed data
        q1_t = transformed_col.quantile(0.25)
        q3_t = transformed_col.quantile(0.75)
        iqr_t = q3_t - q1_t
        lower_bound_t = max(transformed_col.min(), q1_t - 1.5 * iqr_t)
        upper_bound_t = min(transformed_col.max(), q3_t + 1.5 * iqr_t)
        transformed_outliers = ((transformed_col < lower_bound_t) | (transformed_col > upper_bound_t)).sum()
        logger.info(f"{key} transformation --> {transformed_outliers} outliers.")

        # Plot Histogram
        plt.subplot(len(transformations), 2, 2 * i - 1)
        sns.histplot(transformed_col, kde=False, color='gray', binwidth=(transformed_col.max() - transformed_col.min()) / 30)
        plt.axvline(transformed_col.mean(), color='red', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(transformed_col.median(), color='blue', linestyle='dashed', linewidth=1.5, label='Median')
        plt.title(f'Histogram of {column} ({key})')
        plt.legend()

        # Plot Boxplot
        plt.subplot(len(transformations), 2, 2 * i)
        sns.boxplot(x=transformed_col)
        plt.title(f'Boxplot of {column} ({key})')

    plt.tight_layout(pad=3.0)
    plt.show()
    logger.info(f"Completed plotting distributions for {column} with various transformations.")

# Example usage:
# plot_transformations(df, 'your_column_name')