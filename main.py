import streamlit as st
import pandas as pd
import numpy as np
import os
from anthropic import Anthropic
import tempfile
import json
import re
import logging
from typing import Dict, Any, Tuple
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from Claude's response."""
    # Look for code between triple backticks
    code_pattern = r"```python\n(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks found, try to extract just Python-like content
    code_lines = []
    for line in response_text.split('\n'):
        if (line.strip().startswith('def ') or 
            line.strip().startswith('import ') or 
            line.strip().startswith('from ') or
            line.strip().startswith('class ')):
            code_lines.append(line)
            
    return '\n'.join(code_lines) if code_lines else get_default_cleaning_module()

def get_default_cleaning_module() -> str:
    """Return a default cleaning module when code generation fails."""
    return '''
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Default cleaning operations when custom cleaning fails.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Create a copy of the DataFrame
        cleaned_df = df.copy()
        
        # Basic cleaning operations
        for column in cleaned_df.columns:
            # Handle string columns
            if cleaned_df[column].dtype == 'object':
                # Strip whitespace
                if cleaned_df[column].dtype == 'object':
                    cleaned_df[column] = cleaned_df[column].str.strip()
                
                # Convert to lowercase
                cleaned_df[column] = cleaned_df[column].str.lower()
                
            # Handle numeric columns
            elif pd.api.types.is_numeric_dtype(cleaned_df[column]):
                # Replace infinite values with NaN
                cleaned_df[column] = cleaned_df[column].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN with median for numeric columns
                median_value = cleaned_df[column].median()
                cleaned_df[column] = cleaned_df[column].fillna(median_value)
        
        logger.info("Applied default cleaning operations successfully")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error in default cleaning: {e}")
        return df  # Return original DataFrame if cleaning fails
'''

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'files': [],
        'file_states': {},  # Tracks state of each file: {'filename': {'cleaned': bool, 'instructions': str}}
        'cleaned_files': {},
        'current_file': None,
        'cleaned_previews': {},  # Stores preview of cleaned DataFrames
        'original_data': {},     # Stores original DataFrames
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def read_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded file into a pandas DataFrame."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['xlsx', 'xls', 'xlsm']:
            return pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            return pd.read_csv(uploaded_file)
        raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error reading file {uploaded_file.name}: {e}")
        raise

def get_cleaning_code(instructions: str, df_info: Dict[str, Any]) -> str:
    """Generate cleaning code using Claude API based on custom instructions."""
    try:
        client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        
        prompt = f"""You are an expert data cleaning AI assistant. Create a Python function that cleans a pandas DataFrame based on specific instructions.
The function should handle complex, custom cleaning operations beyond basic tasks.

Instructions from user: {instructions}

DataFrame information:
Columns: {df_info['columns']}
Data Types: {df_info['dtypes']}
Sample Data: {df_info['sample_data']}

Requirements:
1. Include all necessary imports at the top
2. Define helper functions if needed
3. Main function must be named 'clean_dataframe' and take a DataFrame as input
4. Use proper error handling for each operation
5. Include logging
6. Handle missing values and type conversions safely
7. Return the cleaned DataFrame even if errors occur

Generate only the Python code without any explanation."""
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        code = extract_code_from_response(response.content[0].text)
        return code if code and 'def clean_dataframe' in code else get_default_cleaning_module()
            
    except Exception as error:
        logger.error(f"Error in get_cleaning_code: {error}")
        return get_default_cleaning_module()

def clean_file(df: pd.DataFrame, cleaning_code: str) -> Tuple[pd.DataFrame, Dict]:
    """Execute the generated cleaning code on the DataFrame and return cleaning summary."""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(cleaning_code)
            temp_file = f.name

        import importlib.util
        spec = importlib.util.spec_from_file_location("cleaning_module", temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'clean_dataframe'):
            raise AttributeError("Cleaning module missing clean_dataframe function")
        
        cleaned_df = module.clean_dataframe(df.copy())
        
        if not isinstance(cleaned_df, pd.DataFrame):
            raise ValueError("Cleaning function must return a DataFrame")
        
        # Generate cleaning summary
        summary = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'changed_columns': [col for col in df.columns if col in cleaned_df.columns and not df[col].equals(cleaned_df[col])],
            'missing_values_before': df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum()
        }
        
        return cleaned_df, summary
        
    except Exception as error:
        logger.error(f"Error during cleaning execution: {error}")
        raise
        
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as error:
                logger.error(f"Error removing temporary file: {error}")

def save_dataframe(df: pd.DataFrame, original_filename: str) -> Tuple[bytes, str]:
    """Save the cleaned DataFrame to an Excel file."""
    try:
        base_name = os.path.splitext(original_filename)[0]
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        buffer.seek(0)
        return buffer.getvalue(), f"{base_name}_cleaned.xlsx"
    except Exception as error:
        logger.error(f"Error saving DataFrame: {error}")
        raise

def display_file_status(filename: str, file_state: Dict):
    """Display status and controls for a single file."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"📄 {filename}")
    
    with col2:
        status = "✅ Cleaned" if file_state.get('cleaned', False) else "⏳ Pending"
        st.write(status)
    
    with col3:
        if file_state.get('cleaned', False):
            if st.button("Reclean", key=f"reclean_{filename}"):
                st.session_state.current_file = filename
                file_state['cleaned'] = False
                st.rerun()

def process_current_file():
    """Process the currently selected file."""
    filename = st.session_state.current_file
    if not filename:
        return
    
    file_state = st.session_state.file_states[filename]
    df = st.session_state.original_data[filename]
    
    st.subheader(f"Processing: {filename}")
    
    st.write("### Original Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    instructions = st.text_area(
        "Cleaning Instructions",
        value=file_state.get('instructions', ''),
        help="""Provide specific instructions for cleaning this data. Examples:
        - Handle missing values in specific columns
        - Convert specific columns to numeric format
        - Standardize text data
        - Remove or flag outliers
        """,
        key=f"instructions_{filename}",
        height=150
    )
    
    if st.button("Process File", key=f"process_{filename}"):
        if not instructions.strip():
            st.warning("⚠️ Please provide cleaning instructions")
            return
            
        with st.spinner("🔄 Generating and applying cleaning operations..."):
            df_info = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head().to_dict()
            }
            
            cleaning_code = get_cleaning_code(instructions, df_info)
            with st.expander("View Generated Cleaning Code"):
                st.code(cleaning_code, language="python")
            
            try:
                cleaned_df, summary = clean_file(df, cleaning_code)
                
                st.write("### Cleaning Summary")
                st.write(f"- Original rows: {summary['original_rows']}")
                st.write(f"- Cleaned rows: {summary['cleaned_rows']}")
                st.write(f"- Modified columns: {', '.join(summary['changed_columns']) if summary['changed_columns'] else 'None'}")
                st.write(f"- Missing values reduced from {summary['missing_values_before']} to {summary['missing_values_after']}")
                
                st.write("### Cleaned Data Preview")
                st.dataframe(cleaned_df.head(), use_container_width=True)
                
                content, output_filename = save_dataframe(cleaned_df, filename)
                
                # Update session state
                file_state['cleaned'] = True
                file_state['instructions'] = instructions
                st.session_state.cleaned_files[filename] = {
                    'content': content,
                    'output_filename': output_filename
                }
                st.session_state.cleaned_previews[filename] = cleaned_df
                
                st.success("✅ File processed successfully!")
                st.download_button(
                    f"📥 Download {output_filename}",
                    content,
                    output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_processed_{filename}"  # Added unique key
                )
                
            except Exception as error:
                st.error(f"Error cleaning file: {error}")
                file_state['cleaned'] = False

def main():
    st.title("AI-Powered Data Cleaning Assistant")
    st.markdown("""
    This tool uses AI to perform custom data cleaning operations on multiple files.
    Upload up to 10 files and provide specific cleaning instructions for each one.
    """)
    
    initialize_session_state()
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload Excel/CSV files (maximum 10 files)",
        type=['xlsx', 'xls', 'csv', 'xlsm'],
        accept_multiple_files=True,
        help="Supported formats: Excel (.xlsx, .xls, .xlsm) and CSV"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("⚠️ Please upload a maximum of 10 files")
            return
            
        # Process new uploads
        new_files = [f for f in uploaded_files if f.name not in st.session_state.file_states]
        for file in new_files:
            try:
                df = read_file(file)
                st.session_state.original_data[file.name] = df
                st.session_state.file_states[file.name] = {
                    'cleaned': False,
                    'instructions': ''
                }
            except Exception as error:
                st.error(f"Error reading {file.name}: {error}")
    
    if not st.session_state.file_states:
        st.info("Please upload one or more files to begin.")
        return
    
    # Display file status and controls
    st.subheader("Files Status")
    for filename, file_state in st.session_state.file_states.items():
        display_file_status(filename, file_state)
    
    # Select file to process
    if st.session_state.current_file is None:
        unprocessed_files = [f for f, state in st.session_state.file_states.items() if not state['cleaned']]
        if unprocessed_files:
            st.session_state.current_file = unprocessed_files[0]
    
    # Process current file
    if st.session_state.current_file:
        process_current_file()
    
    # Download all cleaned files
    if any(state['cleaned'] for state in st.session_state.file_states.values()):
        st.subheader("Download Cleaned Files")
        for filename, file_info in st.session_state.cleaned_files.items():
            st.download_button(
                f"📥 Download {file_info['output_filename']}",
                file_info['content'],
                file_info['output_filename'],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_all_{filename}"  # Added unique key
            )
    
    # Reset button
    if st.button("Start Over", key="reset_button"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()