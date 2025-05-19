# src/app.py
import os
import base64
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, Response, abort, send_from_directory
import json
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from decimal import Decimal, InvalidOperation
from functools import wraps
import datetime
from sqlalchemy import or_ # E-posta ve kullanıcı ID'sine göre arama için

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Path Tanımlamaları
basedir = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(basedir)
INSTANCE_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'instance')
USER_PHOTOS_PATH = os.path.join(INSTANCE_FOLDER_PATH, 'user_photos')
SETTINGS_FILE_PATH = os.path.join(INSTANCE_FOLDER_PATH, 'settings.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, instance_path=INSTANCE_FOLDER_PATH)

# --- Ayarları Yükleme ve Kaydetme Fonksiyonları ---
DEFAULT_SETTINGS = {
    'FACE_RECOGNITION_FEE': '1.50'
}

def load_settings():
    if not os.path.exists(SETTINGS_FILE_PATH):
        print(f"'{SETTINGS_FILE_PATH}' bulunamadı, varsayılan ayarlarla oluşturuluyor.")
        with open(SETTINGS_FILE_PATH, 'w') as f:
            json.dump(DEFAULT_SETTINGS, f, indent=4)
        return DEFAULT_SETTINGS
    try:
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
            updated = False
            for key, value in DEFAULT_SETTINGS.items():
                if key not in settings:
                    settings[key] = value
                    updated = True
            if updated:
                 with open(SETTINGS_FILE_PATH, 'w') as f_update:
                    json.dump(settings, f_update, indent=4)
            return settings
    except (IOError, json.JSONDecodeError) as e:
        print(f"HATA: '{SETTINGS_FILE_PATH}' okunurken veya çözümlenirken sorun oluştu: {e}")
        print("Varsayılan ayarlarla dosya yeniden oluşturuluyor.")
        with open(SETTINGS_FILE_PATH, 'w') as f:
            json.dump(DEFAULT_SETTINGS, f, indent=4)
        return DEFAULT_SETTINGS

def save_settings(settings_data):
    try:
        with open(SETTINGS_FILE_PATH, 'w') as f:
            json.dump(settings_data, f, indent=4)
        app_settings_reloaded = load_settings()
        new_fee_str = app_settings_reloaded.get('FACE_RECOGNITION_FEE', DEFAULT_SETTINGS['FACE_RECOGNITION_FEE'])
        try:
            app.config['FACE_RECOGNITION_FEE'] = Decimal(new_fee_str)
        except InvalidOperation:
            app.config['FACE_RECOGNITION_FEE'] = Decimal(DEFAULT_SETTINGS['FACE_RECOGNITION_FEE'])
            print(f"UYARI: settings.json'daki FACE_RECOGNITION_FEE ('{new_fee_str}') geçersiz, varsayılana dönüldü.")
        print("Ayarlar kaydedildi ve app.config güncellendi.")
        return True
    except IOError as e:
        print(f"HATA: Ayarlar dosyası kaydedilemedi: {e}")
        return False

app_settings = load_settings()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '51614224ab65a418b29e41a41564562fd059d0b27af0e080'
app.config['USER_FACE_RECOGNITION_THRESHOLD'] = 0.5
app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = USER_PHOTOS_PATH
try:
    app.config['FACE_RECOGNITION_FEE'] = Decimal(app_settings.get('FACE_RECOGNITION_FEE', DEFAULT_SETTINGS['FACE_RECOGNITION_FEE']))
except InvalidOperation:
    print(f"UYARI: Başlangıçtaki FACE_RECOGNITION_FEE ('{app_settings.get('FACE_RECOGNITION_FEE')}') geçersiz, varsayılana dönüldü.")
    app.config['FACE_RECOGNITION_FEE'] = Decimal(DEFAULT_SETTINGS['FACE_RECOGNITION_FEE'])

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
    balance = db.Column(db.Numeric(10, 2), nullable=False, default=Decimal('0.00'))
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    transactions = db.relationship('Transaction', backref='user', lazy='dynamic', cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    def __repr__(self):
        return f'<User {self.email}>'

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False, index=True) # index=True eklendi
    transaction_type = db.Column(db.String(50), nullable=False, index=True) # index=True eklendi
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow, index=True) # index=True eklendi
    description = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<Transaction {self.id} {self.transaction_type} {self.amount} by User {self.user_id}>'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def create_json_response(data, status_code=200):
    json_data = json.dumps(data, ensure_ascii=False, default=str)
    return Response(json_data, status=status_code, content_type='application/json; charset=utf-8')

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Bu sayfaya erişim yetkiniz yok.", "danger")
            return redirect(url_for('go_to_dashboard_or_login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def go_to_dashboard_or_login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
# ... (register fonksiyonu aynı) ...
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
            
            new_user = User(email=email, photo_filename=photo_saved_filename, embedding=user_embedding_bytes, balance=Decimal('10.00'))
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            initial_balance_transaction = Transaction(user_id=new_user.id, transaction_type='initial_balance', amount=new_user.balance, description="Yeni kullanıcı başlangıç bakiyesi")
            db.session.add(initial_balance_transaction)
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
# ... (login fonksiyonu aynı) ...
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
# ... (logout fonksiyonu aynı) ...
def logout():
    logout_user()
    flash('Başarıyla çıkış yaptınız.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
# ... (dashboard fonksiyonu aynı) ...
def dashboard():
    return render_template('dashboard.html', 
                           user_email=current_user.email, 
                           current_balance=current_user.balance, 
                           is_admin=current_user.is_admin)

@app.route('/get_balance', methods=['GET'])
@login_required
# ... (get_balance fonksiyonu aynı) ...
def get_balance():
    return create_json_response({'balance': current_user.balance})

@app.route('/add_balance_page', methods=['GET'])
@login_required
# ... (add_balance_page fonksiyonu aynı) ...
def add_balance_page():
    return render_template('add_balance.html')

@app.route('/process_add_balance', methods=['POST'])
@login_required
# ... (process_add_balance fonksiyonu aynı, Transaction logu eklenmişti) ...
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
        transaction = Transaction(user_id=current_user.id, 
                                  transaction_type='bakiye_yukleme', 
                                  amount=amount,
                                  description=f"{amount} birim bakiye eklendi.")
        db.session.add(transaction)
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
# ... (make_payment_page fonksiyonu aynı) ...
def make_payment_page():
    return render_template('make_payment.html', fee=app.config['FACE_RECOGNITION_FEE'])

@app.route('/process_payment_with_face', methods=['POST'])
@login_required
# ... (process_payment_with_face fonksiyonu aynı, Transaction logu eklenmişti ve geliştirildi) ...
def process_payment_with_face():
    if not face_analyzer_app:
        return create_json_response({'error': 'Yüz tanıma servisi aktif değil.', 'status': 'error'}, 503)
    
    data = request.get_json()
    if not data or 'image_data_url' not in data:
        return create_json_response({'error': 'Resim verisi (image_data_url) eksik.', 'status': 'error'}, 400)
    
    image_data_url = data['image_data_url']
    if not image_data_url.startswith('data:image/jpeg;base64,'):
        return create_json_response({'error': 'Geçersiz resim formatı.', 'status': 'error'}, 400)

    transaction_description = ""
    transaction_status_type = "odeme_denemesi" 
    transaction_amount = Decimal('0.00') 

    try:
        base64_image_data = image_data_url.split(',')[1]
        image_data_bytes = base64.b64decode(base64_image_data)
        np_arr = np.frombuffer(image_data_bytes, np.uint8)
        img_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_frame is None:
            transaction_description = "Gönderilen resim verisi çözümlenemedi."
            transaction_status_type = "odeme_hata_resim_cozumlenemedi"
            db.session.add(Transaction(user_id=current_user.id, transaction_type=transaction_status_type, amount=transaction_amount, description=transaction_description))
            db.session.commit()
            return create_json_response({'error': transaction_description, 'status': 'error'}, 400)

        rgb_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
        detected_faces = face_analyzer_app.get(rgb_frame)

        if not detected_faces:
            transaction_description = "Kameradan gönderilen karede yüz bulunamadı."
            transaction_status_type = "odeme_basarisiz_yuz_yok"
            db.session.add(Transaction(user_id=current_user.id, transaction_type=transaction_status_type, amount=transaction_amount, description=transaction_description))
            db.session.commit()
            return create_json_response({'name': 'Unknown', 'score': 0.0, 'message': transaction_description, 'status': 'no_face'})
        
        probe_face = detected_faces[0]
        probe_embedding = probe_face.normed_embedding.reshape(1, -1)
        user_to_verify = current_user 
        
        if not user_to_verify.embedding:
            transaction_description = "Kullanıcının kayıtlı yüz özelliği (embedding) bulunamadı."
            transaction_status_type = "odeme_basarisiz_kayitli_embedding_yok"
            db.session.add(Transaction(user_id=current_user.id, transaction_type=transaction_status_type, amount=transaction_amount, description=transaction_description))
            db.session.commit()
            return create_json_response({'name': 'Unknown', 'score': 0.0, 'message': transaction_description, 'status': 'no_embedding'})

        retrieved_embedding = np.frombuffer(user_to_verify.embedding, dtype=np.float32).reshape(1, -1)
        similarity = cosine_similarity(probe_embedding, retrieved_embedding)[0][0]
        threshold = app.config.get('USER_FACE_RECOGNITION_THRESHOLD', 0.5)
        fee = app.config.get('FACE_RECOGNITION_FEE', Decimal('0.00'))

        if similarity >= threshold: 
            if user_to_verify.balance >= fee:
                user_to_verify.balance -= fee
                transaction_status_type = 'odeme_basarili'
                transaction_amount = -fee
                transaction_description = f"Yüz tanıma ile {fee} birim ödeme yapıldı. Benzerlik: {similarity:.4f}"
                db.session.add(Transaction(user_id=user_to_verify.id, transaction_type=transaction_status_type, amount=transaction_amount, description=transaction_description))
                db.session.commit()
                return create_json_response({'name': user_to_verify.email, 'score': float(similarity), 'message': f'Ödeme başarılı! {fee} birim düşüldü.','new_balance': user_to_verify.balance,'status': 'payment_success'})
            else: 
                transaction_status_type = 'odeme_basarisiz_bakiye_yetersiz'
                transaction_description = f"Yüz tanındı (Benzerlik: {similarity:.4f}) ancak bakiye yetersiz ({user_to_verify.balance}) olduğu için {fee} birim ödeme yapılamadı."
                db.session.add(Transaction(user_id=user_to_verify.id, transaction_type=transaction_status_type, amount=transaction_amount, description=transaction_description))
                db.session.commit()
                return create_json_response({'name': user_to_verify.email, 'score': float(similarity), 'message': 'Yüz tanındı ancak bakiye yetersiz.','current_balance': user_to_verify.balance,'status': 'insufficient_balance'})
        else: 
            transaction_status_type = 'odeme_basarisiz_eslesmedi'
            transaction_description = f"Yüz tanınamadı (kullanıcıyla eşleşmedi). Benzerlik: {similarity:.4f}, Eşik: {threshold}"
            db.session.add(Transaction(user_id=current_user.id, transaction_type=transaction_status_type, amount=transaction_amount, description=transaction_description))
            db.session.commit()
            return create_json_response({'name': 'Unknown', 'score': float(similarity), 'message': 'Yüz tanınamadı (kullanıcıyla eşleşmedi).','status': 'recognition_failed'})

    except Exception as e:
        db.session.rollback() 
        app.logger.error(f"Yüz tanıma ile ödeme hatası: {e}", exc_info=True)
        db.session.add(Transaction(user_id=current_user.id, transaction_type='odeme_hata_sunucu', amount=Decimal('0.00'), description=f"Ödeme sırasında sunucu hatası: {str(e)[:200]}"))
        db.session.commit()
        return create_json_response({'error': f'Ödeme sırasında bir hata oluştu: {str(e)}', 'status': 'error'}, 500)

@app.route('/test_turkce_json')
def test_turkce_json():
    # ... (test_turkce_json fonksiyonu aynı) ...
    print(f"DEBUG: test_turkce_json içinde JSON_AS_ASCII ayarı (config'den): {app.config.get('JSON_AS_ASCII')}")
    data = {"mesaj": "Başarı! Şemsi Paşa pasajında sesi büzüşesiceler."}
    return create_json_response(data)

# --- ADMIN ROTLARI ---
@app.route('/admin')
@login_required
@admin_required 
def admin_dashboard():
    return render_template('admin_dashboard.html', page_title="Admin Ana Sayfa")

@app.route('/admin/users', methods=['GET'])
@login_required
@admin_required
def admin_list_users():
    # ... (admin_list_users fonksiyonu aynı) ...
    page = request.args.get('page', 1, type=int)
    per_page = 10 
    search_query = request.args.get('q', '')
    query = User.query
    if search_query:
        query = query.filter(User.email.ilike(f'%{search_query}%'))
    all_users_pagination = query.order_by(User.id.asc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('admin_users.html', 
                           users=all_users_pagination.items, 
                           pagination=all_users_pagination, 
                           page_title="Kullanıcıları Yönet",
                           search_query=search_query)

@app.route('/admin/user/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_user(user_id):
    # ... (admin_edit_user fonksiyonu aynı) ...
    user_to_edit = db.session.get(User, user_id)
    if not user_to_edit:
        flash('Kullanıcı bulunamadı.', 'danger')
        return redirect(url_for('admin_list_users'))
    if request.method == 'POST':
        new_email = request.form.get('email')
        new_balance_str = request.form.get('balance')
        is_admin_form = request.form.get('is_admin') == 'on'
        new_photo = request.files.get('new_photo')
        if new_email != user_to_edit.email:
            existing_user_with_email = User.query.filter(User.email == new_email, User.id != user_to_edit.id).first()
            if existing_user_with_email:
                flash('Bu e-posta adresi başka bir kullanıcı tarafından kullanılıyor.', 'danger')
                return render_template('admin_edit_user.html', user_to_edit=user_to_edit, page_title=f"Kullanıcıyı Düzenle: {user_to_edit.email}", photo_url=url_for('serve_user_photo', filename=user_to_edit.photo_filename) if user_to_edit.photo_filename else None)
            user_to_edit.email = new_email
        try:
            user_to_edit.balance = Decimal(new_balance_str)
        except InvalidOperation:
            flash('Geçersiz bakiye formatı.', 'danger')
            return render_template('admin_edit_user.html', user_to_edit=user_to_edit, page_title=f"Kullanıcıyı Düzenle: {user_to_edit.email}", photo_url=url_for('serve_user_photo', filename=user_to_edit.photo_filename) if user_to_edit.photo_filename else None)
        user_to_edit.is_admin = is_admin_form
        if new_photo and new_photo.filename != '' and allowed_file(new_photo.filename):
            if user_to_edit.photo_filename:
                old_photo_path = os.path.join(USER_PHOTOS_PATH, user_to_edit.photo_filename)
                if os.path.exists(old_photo_path):
                    try:
                        os.remove(old_photo_path)
                        print(f"Eski fotoğraf silindi: {old_photo_path}")
                    except Exception as e_del:
                        print(f"Eski fotoğraf silinirken hata: {e_del}")
            filename = secure_filename(f"{uuid.uuid4()}_{new_photo.filename}")
            photo_file_path = os.path.join(USER_PHOTOS_PATH, filename)
            new_photo.save(photo_file_path)
            user_to_edit.photo_filename = filename
            print(f"Yeni fotoğraf kaydedildi: {photo_file_path}")
            if face_analyzer_app:
                try:
                    img_for_embedding = cv2.imread(photo_file_path)
                    if img_for_embedding is not None:
                        rgb_img = cv2.cvtColor(img_for_embedding, cv2.COLOR_BGR2RGB)
                        faces = face_analyzer_app.get(rgb_img)
                        if faces and len(faces) == 1:
                            user_to_edit.embedding = faces[0].normed_embedding.tobytes()
                            print(f"Yeni embedding başarıyla çıkarıldı ve güncellendi: {user_to_edit.email}")
                        else:
                            user_to_edit.embedding = None
                            flash('Yeni fotoğrafta yüz bulunamadı/çoklu yüz var, embedding güncellenmedi.', 'warning')
                    else:
                         flash('Yeni fotoğraf işlenirken bir sorun oluştu (okuma), embedding güncellenmedi.', 'error')
                except Exception as e_embed_edit:
                    app.logger.error(f"Kullanıcı düzenlemede embedding hatası: {e_embed_edit}", exc_info=True)
                    flash('Yeni fotoğraf için yüz özellikleri çıkarılırken bir hata oluştu.', 'error')
            else:
                flash('Yüz tanıma servisi aktif değil, embedding güncellenemedi.', 'warning')
        elif new_photo and new_photo.filename != '':
            flash('Geçersiz dosya türü. Lütfen .png, .jpg veya .jpeg uzantılı bir fotoğraf yükleyin.', 'danger')
            return render_template('admin_edit_user.html', user_to_edit=user_to_edit, page_title=f"Kullanıcıyı Düzenle: {user_to_edit.email}", photo_url=url_for('serve_user_photo', filename=user_to_edit.photo_filename) if user_to_edit.photo_filename else None)
        try:
            db.session.commit()
            flash(f"'{user_to_edit.email}' kullanıcısının bilgileri başarıyla güncellendi.", 'success')
            return redirect(url_for('admin_list_users'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Kullanıcı düzenleme (DB commit) hatası: {e}", exc_info=True)
            flash('Kullanıcı güncellenirken bir veritabanı hatası oluştu.', 'danger')
    photo_url = None
    if user_to_edit.photo_filename:
        photo_url = url_for('serve_user_photo', filename=user_to_edit.photo_filename)
    return render_template('admin_edit_user.html', 
                           user_to_edit=user_to_edit, 
                           page_title=f"Kullanıcıyı Düzenle: {user_to_edit.email}",
                           photo_url=photo_url)

@app.route('/user_photo/<filename>')
@login_required 
def serve_user_photo(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    # ... (admin_delete_user fonksiyonu aynı) ...
    user_to_delete = db.session.get(User, user_id)
    if not user_to_delete:
        flash('Silinecek kullanıcı bulunamadı.', 'danger')
        return redirect(url_for('admin_list_users'))
    if current_user.id == user_to_delete.id:
        flash('Kendinizi silemezsiniz.', 'danger')
        return redirect(url_for('admin_list_users'))
    try:
        if user_to_delete.photo_filename:
            photo_path = os.path.join(USER_PHOTOS_PATH, user_to_delete.photo_filename)
            if os.path.exists(photo_path):
                os.remove(photo_path)
                print(f"Fotoğraf silindi: {photo_path}")
        db.session.delete(user_to_delete)
        db.session.commit()
        flash(f"'{user_to_delete.email}' kullanıcısı başarıyla silindi.", 'success')
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Kullanıcı silme hatası: {e}", exc_info=True)
        flash('Kullanıcı silinirken bir hata oluştu.', 'danger')
    return redirect(url_for('admin_list_users'))

@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_settings():
    # ... (admin_settings fonksiyonu aynı) ...
    current_settings = load_settings() 
    if request.method == 'POST':
        new_fee_str = request.form.get('face_recognition_fee')
        try:
            new_fee = Decimal(new_fee_str)
            if new_fee < 0:
                flash("Ücret negatif olamaz.", "danger")
            else:
                current_settings['FACE_RECOGNITION_FEE'] = str(new_fee) 
                if save_settings(current_settings):
                    flash("Ayarlar başarıyla güncellendi.", "success")
                else:
                    flash("Ayarlar dosyaya kaydedilirken bir sorun oluştu.", "danger")
        except InvalidOperation:
            flash("Geçersiz ücret formatı. Lütfen sayısal bir değer girin (örn: 1.50).", "danger")
        return redirect(url_for('admin_settings'))
    current_fee_from_config = app.config.get('FACE_RECOGNITION_FEE', Decimal(DEFAULT_SETTINGS['FACE_RECOGNITION_FEE']))
    return render_template('admin_settings.html', 
                           page_title="Sistem Ayarları", 
                           current_fee=current_fee_from_config)

@app.route('/admin/transactions', methods=['GET']) # methods=['GET'] eklendi
@login_required
@admin_required
def admin_list_transactions():
    page = request.args.get('page', 1, type=int)
    per_page = 15
    
    # Filtreleme parametrelerini al
    search_email = request.args.get('search_email', '').strip()
    transaction_type_filter = request.args.get('transaction_type', '').strip()
    start_date_str = request.args.get('start_date', '').strip()
    end_date_str = request.args.get('end_date', '').strip()

    query = Transaction.query.join(User, Transaction.user_id == User.id) # User tablosuyla join yap

    if search_email:
        query = query.filter(User.email.ilike(f'%{search_email}%'))
    
    if transaction_type_filter:
        query = query.filter(Transaction.transaction_type == transaction_type_filter)
    
    if start_date_str:
        try:
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
            query = query.filter(Transaction.timestamp >= start_date)
        except ValueError:
            flash("Geçersiz başlangıç tarihi formatı. Lütfen YYYY-MM-DD kullanın.", "warning")
            
    if end_date_str:
        try:
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
            # Bitiş tarihini gün sonu olarak almak için (o günkü işlemleri de dahil etmek için)
            end_date = end_date.replace(hour=23, minute=59, second=59)
            query = query.filter(Transaction.timestamp <= end_date)
        except ValueError:
            flash("Geçersiz bitiş tarihi formatı. Lütfen YYYY-MM-DD kullanın.", "warning")

    all_transactions_pagination = query.order_by(Transaction.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('admin_transactions.html', 
                           transactions=all_transactions_pagination.items, 
                           pagination=all_transactions_pagination, 
                           page_title="İşlem Logları",
                           # Filtre değerlerini şablona geri gönder
                           search_email=search_email,
                           transaction_type=transaction_type_filter,
                           start_date=start_date_str,
                           end_date=end_date_str)

if __name__ == '__main__':
    if not os.path.exists(INSTANCE_FOLDER_PATH):
        os.makedirs(INSTANCE_FOLDER_PATH)
    if not os.path.exists(USER_PHOTOS_PATH):
        os.makedirs(USER_PHOTOS_PATH)
        print(f"'{USER_PHOTOS_PATH}' klasörü oluşturuldu.")
    if not os.path.exists(SETTINGS_FILE_PATH):
        load_settings() 
        print(f"'{SETTINGS_FILE_PATH}' oluşturuldu ve varsayılan ayarlar yüklendi.")

    app.run(debug=True, host='0.0.0.0')
