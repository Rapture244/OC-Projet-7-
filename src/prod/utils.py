"""
This module provides utility functions for system environment checks, dataset loading, and DataFrame operations.

Key Features:
1. **System Environment Checks**:
   - `check_full_system_environment`: Logs detailed system information, including Python version, OS details, GPU capabilities,
     and OpenCL platforms/devices.

2. **Dataset Handling**:
   - `load_csv`: Loads a CSV file into a pandas DataFrame after detecting its encoding. Handles common issues such as
     missing files, encoding errors, and empty files.
   - `concat_dataframes`: Concatenates two DataFrames while ensuring column alignment. Adds missing columns with NaN values.
   - `merge_dataframes`: Merges two DataFrames based on a specified key. Supports different types of joins (`left`, `right`, `inner`, `outer`).

3. **Utilities**:
   - `log_section_header`: Logs a visually distinct section header for better log readability.

Dependencies:
- **pandas**: For data manipulation.
- **chardet**: Detects encoding of CSV files.
- **loguru**: Provides structured logging.
- **pyopencl**: Enumerates OpenCL platforms and devices.
- **GPUtil**: Retrieves GPU information.

Notes:
- GPU checks use `nvidia-smi` and `nvcc`. Ensure these tools are installed and accessible in your system PATH for GPU-related checks.
- All functions include robust error handling and logging to assist in debugging and tracking operations.
- Designed to handle common edge cases such as empty DataFrames, missing files, and inconsistent DataFrame columns.

Example Usage:
- **System Checks**:
  - `check_full_system_environment()`: Logs system and GPU details.
- **DataFrame Operations**:
  - `load_csv(file_name="data.csv", parent_path=Path("datasets/"))`: Loads a CSV file from the specified path.
  - `concat_dataframes(base_df=df1, concat_df=df2)`: Concatenates `df1` and `df2` with column alignment.
  - `merge_dataframes(base_df=df1, merge_df=df2, merge_key="id")`: Merges `df1` and `df2` on the `id` column.

"""


# ====================================================== IMPORTS ===================================================== #
# Standard Library Imports
import os
import sys
import platform
import subprocess
from pathlib import Path

# Third-Party Library Imports
import pandas as pd
import chardet
from loguru import logger
import pyopencl as cl
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Typing Imports
from typing import Optional as Opt


# ==================================================================================================================== #
#                                                        SYSTEM                                                        #
# ==================================================================================================================== #
def check_full_system_environment() -> None:
    """
    Logs detailed system and GPU environment information, including:
    - Python version, OS details, architecture, and number of processors.
    - GPU details using CUDA and NVIDIA tools.
    - OpenCL platforms and devices enumeration.
    """
    logger.info("Starting comprehensive system and GPU environment checks...")

    # Log system information
    try:
        python_version: str = sys.version.split()[0]
        os_type: str = platform.system()
        os_version: str = platform.version()
        architecture: str = platform.machine()
        num_processors: Opt[int] = os.cpu_count()

        logger.debug(f"Python Version: {python_version}")
        logger.debug(f"Operating System: {os_type} {os_version}")
        logger.debug(f"Architecture: {architecture}")
        logger.debug(f"Number of Processors: {num_processors if num_processors else 'Unknown'}\n")
    except Exception as e:
        logger.error(f"Failed to retrieve system information: {e}")

    # Check CUDA installation
    try:
        cuda_output: str = subprocess.check_output(["nvcc", "--version"], text=True).strip()
        logger.info(f"CUDA nvcc output:\n{cuda_output}\n")
    except FileNotFoundError:
        logger.warning("CUDA nvcc not found. Ensure CUDA is installed and added to PATH.")
    except Exception as e:
        logger.error(f"Error during CUDA check: {e}")

    # Check GPU compute capability using nvidia-smi
    try:
        nvidia_output: str = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            ).strip()
        logger.info("GPU compute capabilities:\n" + nvidia_output)
    except FileNotFoundError:
        logger.warning("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError:
        logger.warning("Failed to query GPU compute capability via nvidia-smi. Update your drivers.")
    except Exception as e:
        logger.error(f"Error checking GPU compute capability: {e}")

    # Enumerate OpenCL platforms and devices
    logger.info("Starting OpenCL platform and device enumeration...")
    try:
        platforms = cl.get_platforms()
        if not platforms:
            logger.warning("No OpenCL platforms found.")
        else:
            for platform_idx, platform_instance in enumerate(platforms):
                logger.success(f"--- Platform #{platform_idx}: {platform_instance.name} ---")
                logger.debug(f"  Vendor: {platform_instance.vendor}")
                logger.debug(f"  Version: {platform_instance.version}")

                try:
                    devices = platform_instance.get_devices()
                    if not devices:
                        logger.warning(f"No devices found for Platform #{platform_idx} ({platform_instance.name}).")
                        continue

                    for device_idx, device in enumerate(devices):
                        logger.success(f"--- Device #{device_idx}: {device.name} ---")
                        logger.info(f"    Type: {'GPU' if device.type & cl.device_type.GPU else 'CPU'}")
                        logger.info(f"    Compute Units: {device.max_compute_units}")
                        logger.info(f"    Global Memory: {device.global_mem_size / 1024 / 1024:.2f} MB")
                        logger.info(f"    Max Clock Frequency: {device.max_clock_frequency} MHz")
                        logger.info(f"    Max Work Group Size: {device.max_work_group_size}\n")
                except Exception as e:
                    logger.error(f"Error enumerating devices for Platform #{platform_idx}: {e}")
    except Exception as e:
        logger.error(f"An error occurred while enumerating OpenCL platforms: {e}")

    logger.success("Comprehensive system and GPU environment checks completed successfully.")


# ==================================================================================================================== #
#                                                   TERMINAL HEADERS                                                   #
# ==================================================================================================================== #
# rich
console = Console(width=120)


def log_section_header(title: str):
    """
    Logs a visually distinct section header as a boxed panel with Rich.
    """
    # Create a centered Text object for the panel content
    centered_title = Text(title.upper(), style="bold cyan", justify="center")

    # Pass the Text object into the Panel
    panel = Panel(
        centered_title,  # Renderable content
        style="bold cyan",  # Add styling for the panel
        title_align="center",  # Align the border title in the center
        border_style="cyan",  # Border styling
        expand=True,  # Adjusts the panel width to fit the console width
        )
    console.print(panel)


def log_section_subheader(title: str):
    """
    Logs a visually distinct section header with Rich, using a bold, centered title and decorative borders.
    """
    console.rule(title.upper(), style="bold cyan")


# ==================================================================================================================== #
#                                                       DATASETS                                                       #
# ==================================================================================================================== #
def load_csv(file_name: str, parent_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame after detecting the encoding,
    and drop the 'Unnamed: 0' column if it exists.

    Args:
        file_name (str): The file name of the CSV file to load.
        parent_path (Path): Path object representing the base directory for datasets.

    Returns:
        pd.DataFrame: The loaded DataFrame if successful.
    """
    csv_file_path: Path = parent_path / file_name

    try:
        # Detect encoding by reading a sample of the file
        with csv_file_path.open('rb') as file:
            encoding_result = chardet.detect(file.read(10_000))
            file_encoding: str = encoding_result['encoding']

        # Read the CSV file with the detected encoding
        df: pd.DataFrame = pd.read_csv(csv_file_path, encoding=file_encoding)

        # Drop 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        logger.info(
            f"{'Loaded':<10} {file_name:<40} {'Shape:':<10} {str(df.shape):<20} {'Encoding:':<10} {file_encoding}"
            )

        return df

    except FileNotFoundError:
        logger.error(f"FileNotFoundError: '{csv_file_path}' could not be found.")
    except UnicodeDecodeError:
        logger.error(f"UnicodeDecodeError: Could not decode '{csv_file_path}' with detected encoding '{file_encoding}'.")
    except pd.errors.EmptyDataError:
        logger.error(f"EmptyDataError: The file '{csv_file_path}' is empty and cannot be read as a DataFrame.")
    except pd.errors.ParserError:
        logger.error(f"ParserError: The file '{csv_file_path}' could not be parsed correctly. Please check for inconsistencies.")
    except Exception as e:
        logger.exception(f"UnexpectedError: An unexpected error occurred while loading '{csv_file_path}'. Error: {e}")

    # Return None if loading fails
    return None


def concat_dataframes(base_df: pd.DataFrame, concat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate two DataFrames into one, handling cases where columns do not match by adding missing columns as NaN.

    Args:
        base_df (pd.DataFrame): The main DataFrame to which another DataFrame will be concatenated.
        concat_df (pd.DataFrame): The DataFrame to be concatenated to the base DataFrame.

    Returns:
        pd.DataFrame: A DataFrame resulting from concatenating the given DataFrame to the base DataFrame.

    Raises:
        ValueError: If either of the input DataFrames is empty.
    """
    # Validate input DataFrames
    if base_df.empty:
        logger.error("The base DataFrame is empty.")
        raise ValueError("The base DataFrame must not be empty.")

    if concat_df.empty:
        logger.error("The DataFrame to concatenate is empty.")
        raise ValueError("The DataFrame to concatenate must not be empty.")

    # Align columns by adding missing columns with NaN
    all_columns = sorted(set(base_df.columns).union(concat_df.columns))
    base_df = base_df.reindex(columns=all_columns, fill_value=pd.NA)
    concat_df = concat_df.reindex(columns=all_columns, fill_value=pd.NA)

    # Concatenate the DataFrames
    concatenated_df = pd.concat([base_df, concat_df], ignore_index=True)
    logger.info(f"DataFrames concatenated successfully. Shape of the concatenated DataFrame: {concatenated_df.shape}")

    return concatenated_df


def merge_dataframes(base_df: pd.DataFrame, merge_df: pd.DataFrame, merge_key: str, how: str = 'left') -> pd.DataFrame:
    """
    Merge a DataFrame with a base DataFrame based on a specified key.

    Args:
        base_df (pd.DataFrame): The main DataFrame to which another DataFrame will be merged.
        merge_df (pd.DataFrame): The DataFrame to be merged with the base DataFrame.
        merge_key (str): The key on which the merging should be performed.
        how (str, optional): Type of merge to be performed. Default is 'left'.

    Returns:
        pd.DataFrame: A merged DataFrame resulting from the given DataFrames.

    Raises:
        ValueError: If the merge key is not found in either DataFrame.
        ValueError: If the input DataFrames are empty.
    """
    # Validate inputs
    if base_df.empty:
        logger.error("The base DataFrame is empty.")
        raise ValueError("The base DataFrame must not be empty.")

    if merge_df.empty:
        logger.error("The DataFrame to merge is empty.")
        raise ValueError("The DataFrame to merge must not be empty.")

    if merge_key not in base_df.columns:
        logger.error(f"Merge key '{merge_key}' not found in base DataFrame.")
        raise ValueError(f"Merge key '{merge_key}' must exist in the base DataFrame.")

    if merge_key not in merge_df.columns:
        logger.error(f"Merge key '{merge_key}' not found in the DataFrame to merge.")
        raise ValueError(f"Merge key '{merge_key}' must exist in the DataFrame to merge.")

    # Perform the merge
    try:
        logger.info(f"Merging DataFrames with key '{merge_key}' using '{how}' join.")
        merged_df = base_df.merge(merge_df, how=how, on=merge_key)
        logger.success(f"DataFrames merged successfully. Shape of the merged DataFrame: {merged_df.shape}")
    except Exception as e:
        logger.exception(f"An error occurred during merging: {e}")
        raise ValueError(f"Failed to merge DataFrames due to an error: {e}")

    return merged_df



