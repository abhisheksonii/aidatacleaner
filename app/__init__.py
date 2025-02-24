from flask import Flask
from flask_login import LoginManager
from .config import Config
import os

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Create the app instance at module level
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

login_manager.init_app(app)

# Import routes after app is created to avoid circular imports
from .auth import auth
from .routes import main

# Register blueprints
app.register_blueprint(main)
app.register_blueprint(auth)

def create_app():
    return app