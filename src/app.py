# src/app.py
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user # Flask-Login importları

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(basedir), 'instance', 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '51614224ab65a418b29e41a41564562fd059d0b27af0e080' # Kendi SECRET_KEY'iniz

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Flask-Login için LoginManager'ı başlat
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Giriş yapılmamışsa yönlendirilecek sayfa
login_manager.login_message = "Bu sayfayı görüntülemek için lütfen giriş yapın." # Flash mesajı
login_manager.login_message_category = "info" # Flash mesaj kategorisi

# User modelini Flask-Login ile uyumlu hale getiriyoruz (UserMixin ekleyerek)
class User(UserMixin, db.Model): # UserMixin eklendi
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    # UserMixin, get_id() metodunu zaten id üzerinden sağlar.

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'

# Flask-Login için user_loader fonksiyonu
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id)) # SQLAlchemy 2.0+ için db.session.get kullanımı

@app.route('/')
def hello_world():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: # Eğer kullanıcı zaten giriş yapmışsa dashboard'a yönlendir
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else:
            email = request.form.get('email')
            password = request.form.get('password')

        if not email or not password:
            flash('E-posta ve şifre alanları zorunludur!', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Bu e-posta adresi zaten kayıtlı!', 'error')
            return redirect(url_for('register'))

        new_user = User(email=email)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Yeni kullanıcı başarıyla oluşturuldu! Şimdi giriş yapabilirsiniz.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Kullanıcı oluşturulurken bir hata oluştu: {str(e)}', 'error')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: # Eğer kullanıcı zaten giriş yapmışsa dashboard'a yönlendir
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else:
            email = request.form.get('email')
            password = request.form.get('password')

        if not email or not password:
            flash('E-posta ve şifre alanları zorunludur!', 'error')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Geçersiz e-posta veya şifre!', 'error')
            return redirect(url_for('login'))

        login_user(user) # Kullanıcı oturumunu başlat
        flash(f'Hoşgeldin, {user.email}! Giriş başarılı.', 'success')
        
        next_page = request.args.get('next') # Eğer ?next=/dashboard gibi bir yönlendirme varsa
        return redirect(next_page or url_for('dashboard')) # next_page'e veya dashboard'a yönlendir
    
    return render_template('login.html')

# YENİ EKLENEN /logout ROTASI
@app.route('/logout')
@login_required # Bu sayfaya sadece giriş yapmış kullanıcılar erişebilir
def logout():
    logout_user() # Kullanıcı oturumunu sonlandır
    flash('Başarıyla çıkış yaptınız.', 'success')
    return redirect(url_for('login'))

# GÜNCELLENEN /dashboard ROTASI
@app.route('/dashboard')
@login_required # Bu sayfaya sadece giriş yapmış kullanıcılar erişebilir
def dashboard():
    # current_user Flask-Login tarafından sağlanan, o an giriş yapmış kullanıcı nesnesidir.
    return render_template('dashboard.html', user_email=current_user.email)

if __name__ == '__main__':
    instance_path = os.path.join(os.path.dirname(basedir), 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    app.run(debug=True, host='0.0.0.0')