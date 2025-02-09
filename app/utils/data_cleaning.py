# app/utils/data_cleaning.py
import pandas as pd
import numpy as np
import tempfile
import os
import logging
import importlib.util
import re
from typing import Dict, Any, Tuple
from anthropic import Anthropic
from flask import current_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from Claude's response."""
    code_pattern = r"```python\n(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return get_default_cleaning_module()

def get_default_cleaning_module() -> str:
    """Return a default cleaning module when code generation fails."""
    return '''
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Default cleaning operations when custom cleaning fails."""
    try:
        cleaned_df = df.copy()
        
        # Handle string columns
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype == 'object':
                # Convert to string first to avoid .str accessor error
                cleaned_df[column] = cleaned_df[column].astype(str)
                cleaned_df[column] = cleaned_df[column].str.strip()
                cleaned_df[column] = cleaned_df[column].replace('nan', np.nan)
                
            elif pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                cleaned_df[column] = cleaned_df[column].replace([np.inf, -np.inf], np.nan)
                
        # Handle missing values
        for column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            else:
                cleaned_df[column] = cleaned_df[column].fillna('MISSING')
        
        logger.info("Applied default cleaning operations successfully")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error in default cleaning: {e}")
        return df
'''

def get_cleaning_code(instructions: str, df_info: Dict[str, Any]) -> str:
    """Generate cleaning code using Claude API based on custom instructions."""
    try:
        client = Anthropic(api_key=current_app.config['ANTHROPIC_API_KEY'])
        
        prompt = f"""Create a Python function that cleans a pandas DataFrame based on these instructions:

{instructions}

DataFrame info:
Columns: {df_info['columns']}
Types: {df_info['dtypes']}
Sample: {df_info['sample_data']}

Requirements:
1. Include necessary imports
2. Main function must be named 'clean_dataframe'
3. Handle all errors
4. Include logging
5. Handle missing values safely
6. Return DataFrame even if errors occur
7. Convert object columns to string before using str accessor

Generate only the Python code without explanation."""
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        code = extract_code_from_response(response.content[0].text)
        if not code or 'def clean_dataframe' not in code:
            return get_default_cleaning_module()
        return code
            
    except Exception as error:
        logger.error(f"Error in get_cleaning_code: {error}")
        return get_default_cleaning_module()

def process_file(df: pd.DataFrame, instructions: str) -> Tuple[pd.DataFrame, Dict]:
    """Process a file with the given cleaning instructions."""
    temp_file = None
    try:
        # Prepare DataFrame info for code generation
        df_info = {
            'columns': list(df.columns),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
            'sample_data': df.head().to_dict()
        }
        
        # Get and write cleaning code
        cleaning_code = get_cleaning_code(instructions, df_info)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(cleaning_code)
            temp_file = f.name

        # Import and execute cleaning code
        spec = importlib.util.spec_from_file_location("cleaning_module", temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'clean_dataframe'):
            raise AttributeError("Cleaning module missing clean_dataframe function")
        
        # Clean the DataFrame
        cleaned_df = module.clean_dataframe(df.copy())
        
        if not isinstance(cleaned_df, pd.DataFrame):
            raise ValueError("Cleaning function must return a DataFrame")
        
        # Generate summary
        summary = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'changed_columns': [col for col in df.columns if col in cleaned_df.columns and not df[col].equals(cleaned_df[col])],
            'missing_values_before': df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum()
        }
        
        return cleaned_df, summary
        
    except Exception as error:
        logger.error(f"Error during file processing: {error}")
        raise
        
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as error:
                logger.error(f"Error removing temporary file: {error}")