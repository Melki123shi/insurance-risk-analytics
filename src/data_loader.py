"""
Data Loading Utilities for AlphaCare Insurance Risk Analytics
"""

import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_insurance_data(
    file_path: str,
    parse_dates: bool = True,
    clean_column_names: bool = True
) -> pd.DataFrame:
    """
    Load the AlphaCare insurance dataset with proper data types and basic cleaning.
    
    Parameters:
    -----------
    file_path : str
        Path to the insurance data txt file
    parse_dates : bool
        Whether to parse date columns
    clean_column_names : bool
        Whether to standardize column names (remove spaces, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Loaded and preprocessed insurance dataset
    """
    try:
        logger.info(f"Loading data from: {file_path}")

        # Convert to Path object for better handling
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found at: {path}")

        # Load the data
        df = pd.read_csv(path, delimiter='|', low_memory=False)

        logger.info(f"✅ Data loaded successfully! Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        raise


def load_cleaned_data(
    file_path: str,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load the cleaned insurance dataset for modeling.
    
    Parameters:
    -----------
    file_path : str
        Path to the cleaned insurance data csv file
    parse_dates : bool
        Whether to parse date columns
    
    Returns:
    --------
    pd.DataFrame
        Loaded cleaned insurance dataset
    """
    try:
        logger.info(f"Loading cleaned data from: {file_path}")

        # Convert to Path object for better handling
        path = Path(file_path)

        if not path.exists():
            msg = f"Cleaned data file not found at: {path}"
            raise FileNotFoundError(msg)

        # Load the cleaned data
        df = pd.read_csv(path, parse_dates=parse_dates)

        logger.info(f"✅ Cleaned data loaded successfully! Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"❌ Error loading cleaned data: {str(e)}")
        raise