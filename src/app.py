# src/app.py
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash # render_template, redirect, url_for, flash eklendi
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(basedir), 'instance', 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '51614224ab65a418b29e41a41564562fd059d0b27af0e080' #'super_gizli_bir_anahtar_buraya_yazin' # Flash mesajları için SECRET_KEY gerekli! Lütfen bunu değiştirin.

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'

@app.route('/')
def hello_world():
    # Ana sayfayı login sayfasına yönlendirelim veya bir index.html oluşturalım
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST']) # Artık GET isteklerini de kabul ediyor
def register():
    if request.method == 'POST':
        # API'den JSON veya web formundan gelen veriyi al
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else: # Web formundan geliyorsa
            email = request.form.get('email')
            password = request.form.get('password')

        if not email or not password:
            flash('E-posta ve şifre alanları zorunludur!', 'error')
            return redirect(url_for('register')) # Hata mesajıyla kayıt sayfasına geri dön

        if User.query.filter_by(email=email).first():
            flash('Bu e-posta adresi zaten kayıtlı!', 'error')
            return redirect(url_for('register'))

        new_user = User(email=email)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Yeni kullanıcı başarıyla oluşturuldu! Şimdi giriş yapabilirsiniz.', 'success')
            return redirect(url_for('login')) # Başarılı kayıttan sonra giriş sayfasına yönlendir
        except Exception as e:
            db.session.rollback()
            flash(f'Kullanıcı oluşturulurken bir hata oluştu: {str(e)}', 'error')
            return redirect(url_for('register'))
    
    # GET isteği için kayıt formunu göster
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST']) # Artık GET isteklerini de kabul ediyor
def login():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')
        else: # Web formundan geliyorsa
            email = request.form.get('email')
            password = request.form.get('password')

        if not email or not password:
            flash('E-posta ve şifre alanları zorunludur!', 'error')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash('Geçersiz e-posta veya şifre!', 'error')
            return redirect(url_for('login'))

        # Giriş başarılı. İleride buraya session başlatma eklenecek.
        flash(f'Hoşgeldin, {user.email}! Giriş başarılı.', 'success')
        # Şimdilik basit bir dashboard sayfasına yönlendirelim (bu rotayı oluşturacağız)
        return redirect(url_for('dashboard')) 
    
    # GET isteği için giriş formunu göster
    return render_template('login.html')

# Basit bir dashboard sayfası (giriş sonrası yönlendirme için)
@app.route('/dashboard')
def dashboard():
    # Burada normalde @login_required gibi bir decorator ile sadece giriş yapmış kullanıcıların
    # erişebilmesi sağlanır. Şimdilik basit tutuyoruz.
    # Örnek olarak, flash mesajlarını göstermek için bir template render edebiliriz
    # veya sadece bir metin döndürebiliriz.
    # Basit bir dashboard.html oluşturalım.
    # Önce src/templates/dashboard.html dosyasını oluşturmanız gerekir.
    # return render_template('dashboard.html') 
    return "Giriş Başarılı! Burası Dashboard." # Şimdilik basit bir metin


if __name__ == '__main__':
    instance_path = os.path.join(os.path.dirname(basedir), 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' ekledik, aynı ağdaki diğer cihazlardan erişim için (isteğe bağlı)