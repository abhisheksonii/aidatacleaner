from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, UserMixin, current_user
from werkzeug.security import check_password_hash
from .. import login_manager  # Import login_manager from app

auth = Blueprint('auth', __name__)

class User(UserMixin):
    def __init__(self, id):
        self.id = id
        
    @property
    def is_active(self):
        return True
    
    @staticmethod
    def get(user_id):
        if user_id == "efeso":
            return User(user_id)
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == "efeso" and password == "efeso123":
            user = User("efeso")
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.index'))
        else:
            flash('Invalid username or password')
    
    return render_template('auth/login.html')

@auth.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('auth.login'))