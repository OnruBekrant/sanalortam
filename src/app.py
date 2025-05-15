# src/app.py
import os
from flask import Flask, request, jsonify # request ve jsonify eklendi
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash # Şifreleme için eklendi

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(basedir), 'instance', 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False) # nullable=False eklendi, şifre zorunlu

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'

@app.route('/')
def hello_world():
    return 'Merhaba Dünya! Veritabanı yapılandırıldı ve User modeli hazır.'

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json() # Gelen isteğin JSON verisini al

    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'E-posta ve şifre alanları zorunludur!'}), 400

    email = data.get('email')
    password = data.get('password')

    if User.query.filter_by(email=email).first(): # E-posta zaten var mı kontrol et
        return jsonify({'message': 'Bu e-posta adresi zaten kayıtlı!'}), 400

    new_user = User(email=email)
    new_user.set_password(password) # Şifreyi hash'leyerek ata
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'Yeni kullanıcı başarıyla oluşturuldu!'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Kullanıcı oluşturulurken bir hata oluştu.', 'error': str(e)}), 500

if __name__ == '__main__':
    instance_path = os.path.join(os.path.dirname(basedir), 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    app.run(debug=True)