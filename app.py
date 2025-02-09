from app import create_app
import os

# Create uploads directory if it doesn't exist
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Create the Flask application instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=False for production