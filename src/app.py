# src/app.py
import os
import base64 # Base64 veriyi çözmek için eklendi
import uuid   # Benzersiz dosya adları oluşturmak için eklendi
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Proje ana dizinini ve instance klasörünün yolunu belirle
basedir = os.path.abspath(os.path.dirname(__file__)) # Bu src klasörünü işaret eder
PROJECT_ROOT = os.path.dirname(basedir) # Bu ~/projects/sanalortam/ gibi ana proje dizinini işaret eder
INSTANCE_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'instance')
USER_PHOTOS_PATH = os.path.join(INSTANCE_FOLDER_PATH, 'user_photos')

# Flask uygulamasını instance_path ile başlat
# Bu, app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db' gibi göreceli yolların
# instance klasörü içinde çözümlenmesini sağlar.
app = Flask(__name__, instance_path=INSTANCE_FOLDER_PATH)

# Yapılandırmalar
# Veritabanı URI'si artık instance klasörüne göre otomatik olarak ayarlanacak.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '51614224ab65a418b29e41a41564562fd059d0b27af0e080' # Sizin ürettiğiniz SECRET_KEY

# Veritabanı ve Migrasyon Eklentilerini Başlatma
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Flask-Login için LoginManager'ı başlat ve yapılandır
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Bu sayfayı görüntülemek için lütfen giriş yapın."
login_manager.login_message_category = "info"

# Cache Kontrol Başlıklarını Eklemek İçin Fonksiyon
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Veritabanı Modeli
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    photo_filename = db.Column(db.String(255), nullable=True) # Kullanıcının fotoğraf dosyasının adı

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'

# Flask-Login için user_loader callback fonksiyonu
@login_manager.user_loader
def load_user(user_id):
    # SQLAlchemy 2.0+ için db.session.get kullanımı daha modern.
    # Eğer eski bir SQLAlchemy sürümü varsa User.query.get(int(user_id)) kullanılabilir.
    return db.session.get(User, int(user_id))

# Rotalar (Routes)
@app.route('/')
def hello_world():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        photo_saved_filename = None # Kaydedilen fotoğrafın adını tutacak değişken
        try:
            if request.is_json:
                data = request.get_json()
                email = data.get('email')
                password = data.get('password')
                photo_data_url = data.get('photo_data_url')
            else: # Web formundan gelen veri
                email = request.form.get('email')
                password = request.form.get('password')
                photo_data_url = request.form.get('photo_data_url')

            if not email or not password:
                flash('E-posta ve şifre alanları zorunludur!', 'error')
                return redirect(url_for('register'))

            if User.query.filter_by(email=email).first():
                flash('Bu e-posta adresi zaten kayıtlı!', 'error')
                return redirect(url_for('register'))

            # Fotoğraf verisini işle (eğer gönderildiyse)
            if photo_data_url and photo_data_url.startswith('data:image/jpeg;base64,'):
                base64_image_data = photo_data_url.split(',')[1]
                image_data = base64.b64decode(base64_image_data)
                
                if not os.path.exists(USER_PHOTOS_PATH):
                    os.makedirs(USER_PHOTOS_PATH)
                
                photo_saved_filename = f"{uuid.uuid4()}.jpg"
                photo_file_path = os.path.join(USER_PHOTOS_PATH, photo_saved_filename)
                
                with open(photo_file_path, 'wb') as f:
                    f.write(image_data)
                print(f"Fotoğraf kaydedildi: {photo_file_path}")
            elif photo_data_url: # Veri var ama formatı yanlışsa
                print("Uyarı: Fotoğraf verisi alındı ama formatı yanlış veya beklenmedik.")
                flash('Gönderilen fotoğraf formatı uygun değil. Lütfen JPEG formatında bir fotoğraf çekin.', 'error')
                # İsteğe bağlı olarak burada kaydı durdurabilir veya fotoğrafsız devam edebilirsiniz.
                # Şimdilik fotoğrafsız devam etmesine izin vermeyelim, hata versin.
                return redirect(url_for('register'))


            new_user = User(email=email, photo_filename=photo_saved_filename)
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            flash('Yeni kullanıcı başarıyla oluşturuldu! Şimdi giriş yapabilirsiniz.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Kayıt hatası: {e}", exc_info=True) # Sunucu loguna detaylı hata yazdır
            flash(f'Kullanıcı oluşturulurken bir hata oluştu. Lütfen tekrar deneyin.', 'error')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
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

        login_user(user)
        flash(f'Hoşgeldin, {user.email}! Giriş başarılı.', 'success')
        
        next_page = request.args.get('next')
        return redirect(next_page or url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Başarıyla çıkış yaptınız.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user_email=current_user.email)

# Uygulamayı çalıştırma bloğu
if __name__ == '__main__':
    # instance ve user_photos klasörlerinin var olduğundan emin olalım
    # Bu kontrol Flask app başlatılmadan önce yapılmalı.
    # Flask(..., instance_path=...) kullanıldığı için Flask instance klasörünü kendisi yönetir,
    # ama alt klasör olan user_photos'u bizim oluşturmamız gerekebilir.
    if not os.path.exists(USER_PHOTOS_PATH):
        # Önce instance path'in var olup olmadığını kontrol edip oluşturalım,
        # Flask zaten instance_path'i kullanacak.
        if not os.path.exists(INSTANCE_FOLDER_PATH):
             os.makedirs(INSTANCE_FOLDER_PATH)
        os.makedirs(USER_PHOTOS_PATH)
        print(f"'{USER_PHOTOS_PATH}' klasörü oluşturuldu.")
        
    app.run(debug=True, host='0.0.0.0')