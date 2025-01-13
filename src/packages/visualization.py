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
    fig.suptitle('Distribution of TARGET Variable', fontsize=16)

    # Adjust layout and display the plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Log success message and perform garbage collection
    logger.info("TARGET distribution plots displayed successfully.")
    gc.collect()
    # Log success message
    logger.debug("TARGET distribution plots displayed successfully.")



