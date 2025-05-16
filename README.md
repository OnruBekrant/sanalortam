# Gerçek Zamanlı GPU Destekli Yüz Tanıma ve Bakiye Sistemi

Bu proje, Python, Flask, OpenCV, InsightFace ve PyTorch kütüphanelerini kullanarak geliştirilmiş, web tabanlı bir uygulamadır. Kullanıcıların fotoğraflarıyla kayıt olup, yüz tanıma ile kimlik doğrulaması yaparak bakiye sistemi üzerinden işlem yapmalarını sağlar. Uygulama ayrıca bir admin paneli üzerinden kullanıcı ve sistem ayarlarının yönetilmesine olanak tanır.

## 🌟 Temel Özellikler

* **Kullanıcı Yönetimi:**
    * E-posta ve şifre ile güvenli kayıt ve giriş.
    * Kayıt sırasında web kamerasından fotoğraf çekme ve yüz embedding'i çıkarma.
    * Flask-Login ile oturum yönetimi.
* **Yüz Tanıma Çekirdeği:**
    * `InsightFace` kütüphanesi ile yüksek doğruluklu yüz tespiti ve embedding (özellik vektörü) çıkarma.
    * NVIDIA GPU hızlandırması (`ctx_id=0` ile CUDA kullanımı).
    * Kullanıcının kayıtlı yüz embedding'i ile canlı kamera görüntüsündeki yüzün karşılaştırılması.
* **Bakiye ve Ödeme Sistemi:**
    * Her kullanıcı için veritabanında bakiye tutma (`Decimal` tipi ile hassas hesaplama).
    * Yeni kullanıcılara varsayılan bir başlangıç bakiyesi tanımlama.
    * Kullanıcıların (şimdilik admin üzerinden veya basit bir formla) bakiye yükleyebilmesi.
    * Başarılı bir yüz tanıma işlemi sonrası kullanıcının bakiyesinden belirlenen bir ücretin düşülmesi.
* **Admin Paneli:**
    * Özel admin kullanıcısı yetkilendirmesi (veritabanındaki `is_admin` alanı ile).
    * Admin sayfalarına yetkisiz erişimi engelleme.
    * Kayıtlı tüm kullanıcıları listeleme (sayfalamalı) ve e-postaya göre arama.
    * Kullanıcı bilgilerini (e-posta, bakiye, admin durumu) düzenleme.
    * Kullanıcının profil fotoğrafını görme, değiştirme (yeni fotoğraf yükleyince embedding'i de güncelleme) ve silme.
    * Sistem genelindeki ayarları (örneğin, yüz tanıma işlem ücreti) bir arayüzden yönetme (`settings.json` dosyası ile kalıcı saklama).
* **Kullanıcı Arayüzü:**
    * Flask ve Jinja2 şablon motoru ile oluşturulmuş temel web arayüzleri (kayıt, giriş, kullanıcı paneli, bakiye yükleme, yüz tanıma ile ödeme, admin sayfaları).
    * Kullanıcıya işlemler hakkında geri bildirim sağlayan flash mesajları.
    * JavaScript ile dinamik kamera kullanımı ve API etkileşimi.
* **API Endpoint'leri:**
    * Yüz tanıma ve ödeme işlemi için `/process_payment_with_face` (POST).
    * Kullanıcı bakiyesini almak için `/get_balance` (GET).
    * Türkçe karakterlerin doğru gösterimi için JSON yanıtlarında `ensure_ascii=False` kullanımı.

## 🛠️ Kullanılan Teknolojiler

* **Backend:** Python, Flask, Flask-SQLAlchemy (ORM), Flask-Migrate (Veritabanı Migrasyonu), Flask-Login (Oturum Yönetimi)
* **Yüz Tanıma:** OpenCV, InsightFace, NumPy, scikit-learn (cosine\_similarity için)
* **Veritabanı:** SQLite (geliştirme için)
* **Frontend:** HTML, CSS, JavaScript (temel düzeyde)
* **Diğer:** Werkzeug (WSGI, şifre hash'leme)

## ⚙️ Sistem Gereksinimleri

* **İşletim Sistemi:** Linux (Bu proje Linux Mint 22.1 üzerinde geliştirilmiş ve test edilmiştir)
* **NVIDIA Ekran Kartı:** CUDA ve cuDNN destekli bir NVIDIA GPU (InsightFace'in GPU ile verimli çalışması için).
* **NVIDIA Sürücüleri:** GPU'nuzla uyumlu güncel NVIDIA sürücüleri.
* **CUDA Toolkit:** Proje, CUDA 12.x (örneğin 12.9) ile uyumlu PyTorch ve InsightFace kullanılarak geliştirilmiştir.
* **cuDNN Kütüphanesi:** CUDA Toolkit'inizle uyumlu cuDNN kütüphanesi.

## 📦 Proje Kurulum Adımları

1.  **Projeyi Klonlayın (Eğer GitHub'daysa):**
    ```bash
    git clone <projenizin_github_adresi>
    cd <proje_klasor_adi>
    ```

2.  **Python Sanal Ortamı Oluşturun ve Aktifleştirin:**
    Proje ana dizininde (`~/projects/sanalortam` gibi) aşağıdaki komutları çalıştırın:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Gerekli Python Kütüphanelerini Yükleyin:**
    Öncelikle `pip`'i güncelleyin:
    ```bash
    pip install --upgrade pip
    ```
    Ardından, projenizin bağımlılıklarını içeren bir `requirements.txt` dosyası oluşturmanız ve onu kullanmanız önerilir. Bu dosyayı oluşturmak için sanal ortamınız aktifken ve tüm kütüphaneleriniz kurulu iken proje ana dizininde şu komutu çalıştırın:
    ```bash
    pip freeze > requirements.txt
    ```
    Bu `requirements.txt` dosyasını projenize ekleyin. Başka bir ortamda veya başkası projeyi kurarken şu komutla tüm bağımlılıkları yükleyebilir:
    ```bash
    pip install -r requirements.txt
    ```
    Eğer `requirements.txt` dosyanız yoksa, temel kütüphaneleri manuel olarak kurabilirsiniz (versiyon uyumluluklarına dikkat edin):
    ```bash
    pip install Flask Flask-SQLAlchemy Flask-Migrate Flask-Login Werkzeug
    pip install opencv-python insightface numpy scikit-learn
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128) # Veya sisteminizdeki CUDA ile uyumlu en son sürüm
    pip install psycopg2-binary # PostgreSQL için (SQLite kullanılıyorsa şart değil ama Flask-SQLAlchemy bazen isteyebilir)
    # Gerekirse: sudo apt-get install libfreeimage-dev
    ```

4.  **Veritabanını Oluşturun ve Migrasyonları Uygulayın:**
    Proje ana dizinindeyken:
    ```bash
    # Eğer daha önce yapılmadıysa migrasyon ortamını başlatın:
    # flask --app src/app.py db init 
    
    # Model değişikliklerine göre migrasyon betiği oluşturun:
    # flask --app src/app.py db migrate -m "Gerekli model değişiklikleri"
    
    # Migrasyonları veritabanına uygulayın:
    flask --app src/app.py db upgrade
    ```
    Bu komutlar, `instance/app.db` veritabanı dosyasını ve içindeki tabloları oluşturacaktır/güncelleyecektir.

## 🔧 Yapılandırma

Uygulamayı çalıştırmadan önce `src/app.py` dosyasındaki ve `instance/settings.json` dosyasındaki bazı yapılandırma değerlerini kontrol etmeniz/düzenlemeniz gerekebilir:

* **`src/app.py` Dosyasında:**
    * **`app.config['SECRET_KEY']`**: Flask'ın oturum yönetimi ve flash mesajları için kullandığı gizli anahtar. Güvenli ve rastgele bir değerle değiştirilmelidir. Örnek üretim:
        ```python
        # Python yorumlayıcısında:
        # import os
        # os.urandom(24).hex()
        ```
* **`instance/settings.json` Dosyası:**
    * Bu dosya, uygulama ilk kez çalıştığında `src/app.py` tarafından otomatik olarak `instance` klasöründe oluşturulur ve varsayılan ayarları içerir.
    * **`FACE_RECOGNITION_FEE`**: Her başarılı yüz tanıma işlemi için kullanıcı bakiyesinden düşülecek ücret. Bu değer, admin panelindeki "Sistem Ayarları" sayfasından değiştirilebilir.
* **İlk Admin Kullanıcısını Ayarlama:**
    1.  Normal yolla `/register` sayfasından bir kullanıcı kaydedin (örneğin, kendi e-posta adresinizle).
    2.  DB Browser for SQLite gibi bir araçla `instance/app.db` dosyasını açın.
    3.  `user` tablosuna gidin.
    4.  Admin yapmak istediğiniz kullanıcının satırını bulun ve `is_admin` sütunundaki değeri `0`'dan `1`'e (True) manuel olarak değiştirin. Değişiklikleri kaydedin.
* **Galeri ve Fotoğraf Yolları:**
    * Kullanıcıların kayıt sırasında çektikleri fotoğraflar `instance/user_photos/` klasörüne kaydedilir.
    * Yüz tanıma için referans alınacak (eski `GALLERY_PATH` mantığı) fotoğrafların nereye konulacağı şu anki kodda doğrudan bir kullanıcı arayüzüyle yönetilmiyor. Kayıt olan her kullanıcının kendi fotoğrafı referans olarak kullanılıyor.

## 🚀 Uygulamayı Çalıştırma

1.  Yukarıdaki kurulum ve yapılandırma adımlarını tamamladığınızdan emin olun.
2.  Sanal ortamınızın (`venv`) aktif olduğundan emin olun.
3.  Proje ana dizinindeyken (`~/projects/sanalortam`) aşağıdaki komutla Flask geliştirme sunucusunu başlatın:
    ```bash
    python3 src/app.py
    ```
4.  Terminalde "InsightFace modeli başarıyla yüklendi..." ve "Running on http://127.0.0.1:5000" gibi mesajlar görmelisiniz.
5.  Web tarayıcınızı açın ve `http://127.0.0.1:5000/` adresine gidin. Login sayfasına yönlendirileceksiniz.

## 💻 Kullanım Senaryoları

* **Kullanıcı Kaydı:** `/register` sayfasından e-posta, şifre ve web kameranızla fotoğraf çekerek kayıt olun.
* **Giriş/Çıkış:** `/login` sayfasından giriş yapın. `/logout` ile çıkış yapabilirsiniz.
* **Kullanıcı Paneli (Dashboard):** `/dashboard` sayfasında e-postanızı, mevcut bakiyenizi ve (eğer admin iseniz) "Admin Paneli" linkini görebilirsiniz.
* **Bakiye Yükleme:** Dashboard'dan "Bakiye Yükle" linkiyle `/add_balance_page` sayfasına gidip hesabınıza (şimdilik manuel olarak) bakiye ekleyebilirsiniz.
* **Yüz Tanıma ile Ödeme:** Dashboard'dan "Yüz Tanıma ile Ödeme Yap" linkiyle `/make_payment_page` sayfasına gidin. Kamera açılacak ve yüzünüz tanınmaya çalışılacaktır. Başarılı tanıma ve yeterli bakiye durumunda, belirlenen işlem ücreti bakiyenizden düşülür.
* **Admin Paneli:**
    * Admin olarak giriş yaptıktan sonra `/admin` adresine giderek veya dashboard'daki linki kullanarak admin paneline erişin.
    * **Kullanıcıları Yönet (`/admin/users`):** Kullanıcıları listeleyin, e-postaya göre arayın, bilgilerini (e-posta, bakiye, admin durumu) düzenleyin, profil fotoğraflarını güncelleyin veya kullanıcıları silin.
    * **Sistem Ayarları (`/admin/settings`):** Yüz tanıma işlem ücretini değiştirin.

## 📄 `.gitignore` Dosyası Açıklaması

Projenizdeki `.gitignore` dosyası, Git'in hangi dosya ve klasörleri takip etmemesi gerektiğini belirtir. Önerilen temel içerik:

Sanal Ortam/venv/*.pycpycache/Instance Klasörü (Veritabanı, ayarlar, yüklenen fotoğraflar gibi hassas veya büyük dosyalar içerebilir)/instance/Eğer instance/user_photos/ klasörünü de repoya eklemek istemiyorsanız:/instance/user_photos/Ancak, settings.json gibi temel bir ayar dosyasının boş bir örneği repoda olabilir.İndirilen Kurulum Dosyaları (Sistem genelinde kurulurlar)/cuda-keyring_.deb/cudnn-local-repo-.debIDE ve Editör Dosyaları.vscode/.idea/*.swp*.swoSizin belirttiğiniz dosyalar:
* `/cuda-keyring_1.1-1_all.deb`
* `/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb`
* `/venv`

Bu dosyaların `.gitignore` içinde olması doğrudur.

## 🔮 Potansiyel Geliştirmeler ve Gelecek Fikirleri

* **Frontend İyileştirmeleri:** Daha modern ve kullanıcı dostu bir arayüz (Bootstrap, Tailwind CSS veya bir JavaScript framework'ü ile).
* **Gerçek Ödeme Sistemi Entegrasyonu:** İyzico, PayTR, Stripe gibi ödeme ağ geçitleriyle entegrasyon.
* **Detaylı İşlem Logları:** Admin panelinde yapılan tüm ödeme ve bakiye işlemlerinin loglanması.
* **Gelişmiş Admin Özellikleri:** Kullanıcı rollerini daha detaylı yönetme, istatistikler gösterme.
* **Güvenlik Artırımları:** Kapsamlı giriş doğrulama, XSS koruması, rate limiting.
* **Performans Optimizasyonları:** Büyük veri setleri için vektör veritabanları (FAISS vb.) kullanımı.
* **Asenkron İşlemler:** Uzun sürebilecek işlemler (embedding çıkarma, yoğun API istekleri) için Celery gibi araçlarla asenkron görev kuyrukları kullanma.
* **Docker Desteği:** Uygulamayı ve bağımlılıklarını Docker ile paketleyerek dağıtımı kolaylaştırma.
* **Mobil Uygulama:** Backend API'lerini kullanarak bir mobil uygulama geliştirme.

---

Bu README dosyasını projenizin kök dizinine (`~/projects/sanalortam/README.md`) kaydedebilirsiniz. Umarım bu detaylı dosya, projenizin anlaşılmasına ve geliştirilmesine yardımcı olur!
