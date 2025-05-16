# src/app.py
import os
import base64
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, Response
import json
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from decimal import Decimal, InvalidOperation # Bakiye işlemleri için Decimal kullanımı

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
# import platform as pf # Şu anda kullanılmıyor

# Path Tanımlamaları
basedir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(basedir)
INSTANCE_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'instance')
USER_PHOTOS_PATH = os.path.join(INSTANCE_FOLDER_PATH, 'user_photos')

app = Flask(__name__, instance_path=INSTANCE_FOLDER_PATH)

# Yapılandırmalar
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '51614224ab65a418b29e41a41564562fd059d0b27af0e080'
app.config['USER_FACE_RECOGNITION_THRESHOLD'] = 0.5
app.config['JSON_AS_ASCII'] = False
app.config['FACE_RECOGNITION_FEE'] = Decimal('1.50') # Yüz tanıma ücreti (örneğin 1.50 birim)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Bu sayfayı görüntülemek için lütfen giriş yapın."
login_manager.login_message_category = "info"

print("InsightFace modeli yükleniyor... Bu işlem biraz zaman alabilir.")
try:
    face_analyzer_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    face_analyzer_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace modeli başarıyla yüklendi (GPU kullanılıyor).")
except Exception as e:
    print(f"HATA: InsightFace modeli yüklenirken sorun oluştu: {e}")
    face_analyzer_app = None

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    photo_filename = db.Column(db.String(255), nullable=True)
    embedding = db.Column(db.LargeBinary, nullable=True)
    balance = db.Column(db.Numeric(10, 2), nullable=False, default=Decimal('0.00')) # YENİ: Kullanıcı bakiyesi

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    def __repr__(self):
        return f'<User {self.email}>'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def create_json_response(data, status_code=200):
    json_data = json.dumps(data, ensure_ascii=False, default=str) # Decimal için default=str eklendi
    return Response(json_data, status=status_code, content_type='application/json; charset=utf-8')

@app.route('/')
def go_to_dashboard_or_login(): # Fonksiyon adı daha açıklayıcı hale getirildi
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        photo_saved_filename = None
        user_embedding_bytes = None
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            photo_data_url = request.form.get('photo_data_url')
            if not email or not password:
                flash('E-posta ve şifre alanları zorunludur!', 'error')
                return redirect(url_for('register'))
            if User.query.filter_by(email=email).first():
                flash('Bu e-posta adresi zaten kayıtlı!', 'error')
                return redirect(url_for('register'))
            if photo_data_url and photo_data_url.startswith('data:image/jpeg;base64,'):
                base64_image_data = photo_data_url.split(',')[1]
                image_data_bytes = base64.b64decode(base64_image_data)
                if not os.path.exists(USER_PHOTOS_PATH):
                    os.makedirs(USER_PHOTOS_PATH)
                photo_saved_filename = f"{uuid.uuid4()}.jpg"
                photo_file_path = os.path.join(USER_PHOTOS_PATH, photo_saved_filename)
                with open(photo_file_path, 'wb') as f:
                    f.write(image_data_bytes)
                print(f"Fotoğraf kaydedildi: {photo_file_path}")
                if face_analyzer_app:
                    img_for_embedding = cv2.imread(photo_file_path)
                    if img_for_embedding is not None:
                        rgb_img = cv2.cvtColor(img_for_embedding, cv2.COLOR_BGR2RGB)
                        faces = face_analyzer_app.get(rgb_img)
                        if faces and len(faces) == 1:
                            user_embedding_bytes = faces[0].normed_embedding.tobytes()
                            print(f"Embedding başarıyla çıkarıldı: {email}")
                        elif faces and len(faces) > 1:
                            flash('Fotoğrafta birden fazla yüz algılandı...', 'warning')
                        else:
                            flash('Fotoğrafta yüz algılanamadı...', 'warning')
                    else:
                        flash('Fotoğraf işlenirken bir sorun oluştu (okuma).', 'error')
                else:
                    flash('Yüz tanıma servisi aktif değil, embedding oluşturulamadı.', 'warning')
            elif photo_data_url:
                flash('Gönderilen fotoğraf formatı uygun değil...', 'error')
                return redirect(url_for('register'))
            # Fotoğraf zorunlu değilse ve gönderilmediyse embedding None olacak
            
            new_user = User(email=email, photo_filename=photo_saved_filename, embedding=user_embedding_bytes, balance=Decimal('10.00')) # Yeni kullanıcıya 10 birim bakiye
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Yeni kullanıcı başarıyla oluşturuldu! Hesabınıza 10 birim bakiye eklendi. Şimdi giriş yapabilirsiniz.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Kayıt hatası: {e}", exc_info=True)
            flash(f'Kullanıcı oluşturulurken bir hata oluştu. Lütfen tekrar deneyin.', 'error')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
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
    return render_template('dashboard.html', user_email=current_user.email, current_balance=current_user.balance)

# --- YENİ API ENDPOINT'LERİ ---
@app.route('/get_balance', methods=['GET'])
@login_required
def get_balance():
    return create_json_response({'balance': current_user.balance})

@app.route('/add_balance_page', methods=['GET'])
@login_required
def add_balance_page():
    return render_template('add_balance.html') # Bu HTML dosyasını oluşturacağız

@app.route('/process_add_balance', methods=['POST'])
@login_required
def process_add_balance():
    try:
        amount_str = request.form.get('amount')
        if not amount_str:
            flash('Lütfen bir miktar girin.', 'error')
            return redirect(url_for('add_balance_page'))
        
        amount = Decimal(amount_str)
        if amount <= 0:
            flash('Lütfen pozitif bir miktar girin.', 'error')
            return redirect(url_for('add_balance_page'))

        current_user.balance += amount
        db.session.commit()
        flash(f'{amount} birim başarıyla hesabınıza eklendi. Yeni bakiyeniz: {current_user.balance}', 'success')
        return redirect(url_for('dashboard'))
    except InvalidOperation:
        flash('Geçersiz miktar formatı. Lütfen sayısal bir değer girin (örn: 10.50).', 'error')
        return redirect(url_for('add_balance_page'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Bakiye yükleme hatası: {e}", exc_info=True)
        flash('Bakiye yüklenirken bir hata oluştu.', 'error')
        return redirect(url_for('add_balance_page'))

@app.route('/make_payment_page', methods=['GET'])
@login_required
def make_payment_page():
    return render_template('make_payment.html', fee=app.config['FACE_RECOGNITION_FEE']) # Bu HTML dosyasını oluşturacağız

@app.route('/process_payment_with_face', methods=['POST'])
@login_required # Bu endpoint'i artık korumalı yapıyoruz
def process_payment_with_face():
    if not face_analyzer_app:
        return create_json_response({'error': 'Yüz tanıma servisi aktif değil.', 'status': 'error'}, 503)
    
    data = request.get_json()
    if not data or 'image_data_url' not in data:
        return create_json_response({'error': 'Resim verisi (image_data_url) eksik.', 'status': 'error'}, 400)
    
    image_data_url = data['image_data_url']
    if not image_data_url.startswith('data:image/jpeg;base64,'):
        return create_json_response({'error': 'Geçersiz resim formatı.', 'status': 'error'}, 400)

    try:
        base64_image_data = image_data_url.split(',')[1]
        image_data_bytes = base64.b64decode(base64_image_data)
        np_arr = np.frombuffer(image_data_bytes, np.uint8)
        img_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_frame is None:
            return create_json_response({'error': 'Resim verisi çözümlenemedi.', 'status': 'error'}, 400)

        rgb_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
        detected_faces = face_analyzer_app.get(rgb_frame)

        if not detected_faces:
            return create_json_response({'name': 'Unknown', 'score': 0.0, 'message': 'Karede yüz bulunamadı.', 'status': 'no_face'})
        
        probe_face = detected_faces[0] # İlk tespit edilen yüzü al
        probe_embedding = probe_face.normed_embedding.reshape(1, -1)

        # Karşılaştırılacak kullanıcı şu anki giriş yapmış kullanıcı olmalı
        user_to_verify = current_user 
        if not user_to_verify.embedding:
            return create_json_response({'name': 'Unknown', 'score': 0.0, 'message': 'Kullanıcının kayıtlı yüz özelliği bulunamadı.', 'status': 'no_embedding'})

        retrieved_embedding = np.frombuffer(user_to_verify.embedding, dtype=np.float32).reshape(1, -1)
        
        similarity = cosine_similarity(probe_embedding, retrieved_embedding)[0][0]
        threshold = app.config.get('USER_FACE_RECOGNITION_THRESHOLD', 0.5)
        fee = app.config.get('FACE_RECOGNITION_FEE', Decimal('0.00'))

        if similarity >= threshold: # Yüz tanındı (giriş yapmış kullanıcı ile eşleşti)
            if user_to_verify.balance >= fee:
                user_to_verify.balance -= fee
                db.session.commit()
                return create_json_response({
                    'name': user_to_verify.email, 
                    'score': float(similarity), 
                    'message': f'Ödeme başarılı! {fee} birim düşüldü.',
                    'new_balance': user_to_verify.balance,
                    'status': 'payment_success'
                })
            else:
                return create_json_response({
                    'name': user_to_verify.email, 
                    'score': float(similarity), 
                    'message': 'Yüz tanındı ancak bakiye yetersiz.',
                    'current_balance': user_to_verify.balance,
                    'status': 'insufficient_balance'
                })
        else: # Yüz tanınamadı (giriş yapmış kullanıcı ile eşleşmedi)
            return create_json_response({
                'name': 'Unknown', 
                'score': float(similarity), 
                'message': 'Yüz tanınamadı (kullanıcıyla eşleşmedi).',
                'status': 'recognition_failed'
            })

    except Exception as e:
        app.logger.error(f"Yüz tanıma ile ödeme hatası: {e}", exc_info=True)
        return create_json_response({'error': f'Ödeme sırasında bir hata oluştu: {str(e)}', 'status': 'error'}, 500)


if __name__ == '__main__':
    if not os.path.exists(INSTANCE_FOLDER_PATH):
        os.makedirs(INSTANCE_FOLDER_PATH)
    if not os.path.exists(USER_PHOTOS_PATH):
        os.makedirs(USER_PHOTOS_PATH)
        print(f"'{USER_PHOTOS_PATH}' klasörü oluşturuldu.")
    app.run(debug=True, host='0.0.0.0')
