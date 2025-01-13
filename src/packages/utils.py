# Standard library imports
import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional as Opt

# Third-party library imports
import pandas as pd
import chardet
from loguru import logger
import GPUtil
import pyopencl as cl

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
    Concatenate two DataFrames into one.

    Args:
        base_df (pd.DataFrame): The main DataFrame to which another DataFrame will be concatenated.
        concat_df (pd.DataFrame): The DataFrame to be concatenated to the base DataFrame.

    Returns:
        pd.DataFrame: A DataFrame resulting from concatenating the given DataFrame to the base DataFrame.

    Raises:
        ValueError: If either of the input DataFrames is empty.
        ValueError: If the columns of the DataFrames do not match.
    """
    # Validate input DataFrames
    if base_df.empty:
        logger.error("The base DataFrame is empty.")
        raise ValueError("The base DataFrame must not be empty.")

    if concat_df.empty:
        logger.error("The DataFrame to concatenate is empty.")
        raise ValueError("The DataFrame to concatenate must not be empty.")

    if not base_df.columns.equals(concat_df.columns):
        logger.error("The columns of the DataFrames do not match.")
        raise ValueError("Both DataFrames must have the same columns to concatenate.")

    # Concatenate the DataFrames
    concatenated_df = pd.concat([base_df, concat_df], ignore_index=True)
    logger.info(f"DataFrames concatenated successfully. Shape of the concatenated DataFrame: {concatenated_df.shape}")

    return concatenated_df





