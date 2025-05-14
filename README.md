# Gerçek Zamanlı GPU Destekli Yüz Tanıma Uygulaması

Bu proje, Python, OpenCV, InsightFace ve PyTorch kütüphanelerini kullanarak bir kamera akışından gerçek zamanlı olarak yüzleri tespit eden ve tanıyan bir uygulamadır. Tanıma işlemi, önceden tanımlanmış bir fotoğraf galerisindeki kişilerle karşılaştırma yapılarak gerçekleştirilir ve GPU hızlandırmasından faydalanır.

## 🌟 Temel Özellikler

* **Gerçek Zamanlı Yüz Tespiti:** Kamera akışındaki yüzleri anlık olarak bulur.
* **Yüz Tanıma:** Tespit edilen yüzleri, bir galerideki bilinen kişilerin yüzleriyle karşılaştırarak kimliklerini belirler.
* **GPU Hızlandırma:** `InsightFace` ve `PyTorch` aracılığıyla NVIDIA GPU'larını kullanarak tespit ve tanıma işlemlerini hızlandırır.
* **Esnek Galeri Yönetimi:** Tanınacak kişilerin fotoğraflarını kolayca bir klasöre ekleyerek galeri oluşturma.
* **Yapılandırılabilir Eşik Değeri:** Yüz tanıma hassasiyetini ayarlamak için bir benzerlik eşiği.
* **Çoklu Kamera Desteği:** Dahili web kamerası veya IP kamera (örneğin telefon kamerası) kullanabilme.

## 🛠️ Gereksinimler ve Kurulum

Bu projeyi çalıştırmak için aşağıdaki yazılım ve kütüphanelerin sisteminizde kurulu olması gerekmektedir.

### 1. Sistem Gereksinimleri

* **İşletim Sistemi:** Linux (Bu proje Linux Mint 22.1 üzerinde geliştirilmiş ve test edilmiştir)
* **NVIDIA Ekran Kartı:** CUDA ve cuDNN destekli bir NVIDIA GPU.
* **NVIDIA Sürücüleri:** GPU'nuzla uyumlu güncel NVIDIA sürücüleri.
* **CUDA Toolkit:** Proje, CUDA 12.x (örneğin 12.9) ile uyumlu PyTorch kullanılarak geliştirilmiştir. Sisteminizde uyumlu bir CUDA Toolkit sürümünün kurulu olması önerilir.
    * `nvcc --version` komutu ile CUDA Toolkit versiyonunuzu kontrol edebilirsiniz.
* **cuDNN Kütüphanesi:** CUDA Toolkit'inizle uyumlu cuDNN kütüphanesi.
    * Bu proje, PyTorch'un kendi içinde getirdiği cuDNN kütüphanelerini de kullanabilir.

### 2. Python Kurulumu

* **Python:** Python 3.9 veya üzeri (Proje Python 3.12.3 ile test edilmiştir).
* **pip:** Python paket yöneticisi.
* **venv:** Sanal ortam oluşturmak için Python'un dahili modülü.

### 3. Proje Kurulum Adımları

1.  **Projeyi Klonlayın (Eğer GitHub'daysa):**
    ```bash
    git clone <projenizin_github_adresi>
    cd <proje_klasor_adi>
    ```

2.  **Sanal Ortam Oluşturun ve Aktifleştirin:**
    Proje ana dizininde (`~/projects/sanalortam` gibi) aşağıdaki komutları çalıştırın:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    Terminal komut satırınızın başında `(venv)` ifadesini görmelisiniz.

3.  **Gerekli Python Kütüphanelerini Yükleyin:**
    Aşağıdaki komutlarla tüm bağımlılıkları sanal ortamınıza kurun:
    ```bash
    pip install --upgrade pip
    pip install opencv-python opencv-contrib-python
    pip install numpy matplotlib scikit-learn
    # PyTorch (CUDA 12.8 uyumlu - sisteminizdeki sürücü ve toolkit ile kontrol edin)
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
    # Diğer kütüphaneler
    pip install onnxruntime-gpu insightface
    # Gerekirse FreeImage (mnistCUDNN örneği için kurulmuştu, bu proje için doğrudan gerekli olmayabilir)
    # sudo apt-get install libfreeimage-dev
    ```
    *Not: `insightface` ve `onnxruntime-gpu` kurulumları sırasında gerekli CUDA/cuDNN uyumlulukları için kendi belgelerini kontrol etmek faydalı olabilir.*

## ⚙️ Yapılandırma

Uygulamayı çalıştırmadan önce `src/algo.py` (veya ana betik dosyanızın adı neyse) dosyasındaki aşağıdaki yapılandırma değişkenlerini kendi sisteminize göre düzenlemeniz gerekebilir:

* **`GALLERY_PATH`**:
    * Tanınacak kişilerin fotoğraflarının bulunduğu klasörün tam yolu.
    * Varsayılan: `/home/onur/projects/sanalortam/data`
    * Bu klasörün içine, her bir kişi için `.jpg`, `.jpeg` veya `.png` formatında bir fotoğraf koyun. Dosya adı, kişinin ismi olmalıdır (örneğin, `Onur_Yilmaz.jpg`).
* **`THRESHOLD`**:
    * Yüz tanıma için kosinüs benzerliği eşik değeri. Bu değerin üzerindeki benzerlikler aynı kişi olarak kabul edilir.
    * Varsayılan: `0.5` (İhtiyaca göre ayarlanabilir).
* **`CAMERA_ID`**:
    * Kullanılacak kamera kaynağı.
    * Dahili web kamerası için: `0` (veya `1`, `2`, ... sistemdeki kamera sayısına göre).
    * IP kamera (örneğin telefon kamerası) için: `"http://<IP_ADRESİ>:<PORT>/video"` veya `"rtsp://<IP_ADRESİ>:<PORT>/..."` formatında URL.
    * Varsayılan: `0`

## 🚀 Kullanım

1.  Yukarıdaki kurulum ve yapılandırma adımlarını tamamladığınızdan emin olun.
2.  Sanal ortamınızın (`venv`) aktif olduğundan emin olun.
3.  Proje ana dizinindeyken aşağıdaki komutla uygulamayı çalıştırın:
    ```bash
    python3 src/algo.py
    ```
    (Eğer betik dosyanızın adı farklıysa, `algo.py` yerine o adı kullanın.)

4.  Uygulama başladığında:
    * Terminalde galeriden yüklenen kişiler hakkında bilgi mesajları göreceksiniz.
    * `insightface` modelleri ilk çalıştırmada indirilebilir (internet bağlantısı gerektirir).
    * Bir pencere açılarak kamera görüntüsü gösterilecektir.
    * Kamera karşısına geçtiğinizde, yüzünüzün etrafında bir kutu ve tanınan kişinin adı (veya "Unknown") ile benzerlik skoru görüntülenecektir.
    * Uygulamadan çıkmak için kamera penceresi aktifken klavyeden `q` tuşuna basın.

## 📄 `.gitignore` Dosyası Açıklaması

Projenizdeki `.gitignore` dosyası, Git versiyon kontrol sisteminin hangi dosya ve klasörleri takip etmemesi gerektiğini belirtir. Bu, gereksiz dosyaların veya hassas bilgilerin depoya (repository) gönderilmesini engeller.

Mevcut `.gitignore` içeriğiniz:
/cuda-keyring_1.1-1_all.deb/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb/venv
* **/cuda-keyring_1.1-1_all.deb**: NVIDIA CUDA deposu için anahtar dosyasının indirilmiş bir kopyası. Bu, her geliştiricinin kendi sistemine kurması gereken bir dosyadır ve projeyle birlikte dağıtılmasına gerek yoktur.
* **/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb**: NVIDIA cuDNN yerel depo dosyasının indirilmiş bir kopyası. Benzer şekilde, bu da sisteme özgü bir kurulum dosyasıdır.
* **/venv**: Python sanal ortam klasörü. Bu klasör, projeye özgü Python yorumlayıcısını ve kurulan tüm kütüphaneleri içerir. Boyutu büyük olabilir ve içeriği `pip install -r requirements.txt` (eğer bir `requirements.txt` dosyanız varsa) komutuyla her sistemde yeniden oluşturulabilir. Bu nedenle, sanal ortam klasörleri genellikle `.gitignore` dosyasına eklenir.

**Öneri:** Projenizin bağımlılıklarını yönetmek için bir `requirements.txt` dosyası oluşturmanız faydalı olacaktır:
```bash
# (venv) aktifken proje ana dizininde:
pip freeze > requirements.txt
Bu requirements.txt dosyasını Git'e ekleyebilirsiniz. Böylece başkaları (veya siz farklı bir sistemde) projeyi kurarken pip install -r requirements.txt komutuyla tüm gerekli Python paketlerini kolayca yükleyebilir. Eğer requirements.txt eklerseniz, .gitignore dosyasının bu dosyayı görmezden gelmediğinden emin olun.🔮 Potansiyel Geliştirmeler ve Gelecek FikirleriYeni Kişi Ekleme Arayüzü: Uygulama çalışırken yeni kişileri galeriye kolayca eklemek için bir arayüz.Kayıt ve Raporlama: Tanınan kişilerin zaman damgalarıyla birlikte bir veritabanına veya dosyaya kaydedilmesi.Performans Optimizasyonları: Daha büyük galeriler veya daha yüksek çözünürlüklü akışlar için optimizasyonlar (örneğin, kare atlama, daha optimize model kullanımı).Web Arayüzü: Sonuçları bir web arayüzü üzerinden gösterme.Docker Desteği: Uygulamayı ve bağımlılıklarını bir Docker konteynerine paketleyerek dağıtımı kolaylaştırma.Bu README dosyasını pro
