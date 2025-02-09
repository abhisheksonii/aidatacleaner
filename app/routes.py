from flask import Blueprint, render_template, request, jsonify, send_file, url_for, current_app
from flask_login import login_required, current_user
from .utils.data_cleaning import process_file
import pandas as pd
import numpy as np
import io
import os
import logging
import datetime
import json

logger = logging.getLogger(__name__)
main = Blueprint('main', __name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def safe_serialize(value):
    """Safely serialize any value to JSON-compatible format."""
    if pd.isna(value):
        return None
    elif isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    elif isinstance(value, (np.int64, np.int32)):
        return int(value)
    elif isinstance(value, np.float64):
        return float(value)
    elif isinstance(value, str):
        return value
    return str(value)

@main.route('/')
@login_required
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files.'}), 400

        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(current_app.root_path, '..', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        try:
            if file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400

        # Prepare preview data
        preview_data = {}
        for column in df.columns:
            series = df[column].head()
            preview_data[column] = {
                str(i): safe_serialize(val) 
                for i, val in enumerate(series)
            }

        # Convert dtypes to strings
        dtypes_dict = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

        response_data = {
            'preview': preview_data,
            'columns': list(map(str, df.columns)),
            'dtypes': dtypes_dict
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main.route('/process', methods=['POST'])
@login_required
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        instructions = request.form.get('instructions', '')

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not instructions:
            return jsonify({'error': 'No cleaning instructions provided'}), 400

        # Read the file
        try:
            if file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400

        # Process the file
        try:
            cleaned_df, summary = process_file(df, instructions)
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 400

        # Save to Excel
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                cleaned_df.to_excel(writer, index=False)
            output.seek(0)
        except Exception as e:
            return jsonify({'error': f'Error saving file: {str(e)}'}), 400

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'cleaned_{file.filename.rsplit(".", 1)[0]}.xlsx'
        )

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': str(e)}), 400