import os
try:
    import cv2
except ImportError:
    raise ImportError("cv2 modülü bulunamadı. 'pip install opencv-python' ile yükleyin.")

import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import platform # Sistem bilgilerini almak için eklendi

# =====================
# Configuration
# =====================
GALLERY_PATH = "/home/onur/projects/sanalortam/data"  # Fotoğrafların doğrudan bulunduğu dizin
THRESHOLD = 0.5 # Yüz tanıma için benzerlik eşiği

# Kamera Ayarları:
# Dahili web kamerası için: CAMERA_ID = 0
# Telefonunuzdaki IP kamera uygulamasından aldığınız URL için:
# CAMERA_ID = "http://<telefonunuzun_ip_adresi>:<port>/video" # Örnek MJPEG URL
# CAMERA_ID = "rtsp://<telefonunuzun_ip_adresi>:<port>/..." # Örnek RTSP URL
CAMERA_ID = 0 # Varsayılan olarak dahili kamera
# Örnek IP Kamera URL'leri (kendi telefonunuzdan aldığınızla değiştirin):
# CAMERA_ID = "http://192.168.1.100:8080/video"
# CAMERA_ID = "https://192.168.122.106/video" # Bu HTTPS ise OpenCV'nin SSL ile derlenmiş olması gerekebilir.

WINDOW_NAME = "FaceRec"

# =====================
# Fonksiyonlar
# =====================
def load_gallery_embeddings(app, gallery_path):
    """
    Belirtilen yoldaki fotoğraflardan yüzleri yükler ve embedding'lerini çıkarır.
    """
    if not os.path.isdir(gallery_path):
        raise FileNotFoundError(f"Galeri yolu '{gallery_path}' bulunamadı.")

    names = []
    embeddings = []
    print(f"\n[Galeri Yükleniyor: {gallery_path}]")
    for img_name in sorted(os.listdir(gallery_path)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(gallery_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Uyarı: '{img_path}' okunamadı.")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Insightface RGB formatını bekler
        faces = app.get(rgb) # Yüzleri tespit et ve embedding'leri al
        if faces:
            # Galerideki her fotoğrafta sadece bir yüz olduğunu varsayıyoruz
            name = os.path.splitext(img_name)[0]  # Dosya adından kişi ismini çıkar
            names.append(name)
            embeddings.append(faces[0].normed_embedding) # İlk bulunan yüzün embedding'i
            print(f"  Yüklendi: {name}")
        else:
            print(f"  Uyarı: '{img_name}' içinde yüz bulunamadı.")

    if embeddings:
        embeddings = np.vstack(embeddings)
        print(f"{len(names)} kişi galeriden başarıyla yüklendi.")
    else:
        embeddings = np.zeros((0, 512)) # Embedding boyutu (genellikle 512)
        print("Uyarı: Galeriden hiç yüz yüklenemedi.")
    return names, gallery_embeddings

# =====================
# Ana Uygulama
# =====================
if __name__ == "__main__":
    print("--- Yüz Tanıma Uygulaması Başlatılıyor ---")
    try:
        # OpenCV GUI desteği kontrolü (önemli)
        build_info = cv2.getBuildInformation()
        gui_support = "YES" # Varsayılan olarak GUI desteği var kabul edelim
        if 'GUI:' in build_info:
            gui_line = [line for line in build_info.splitlines() if 'GUI:' in line][0]
            if 'NO' in gui_line or 'OFF' in gui_line: # Farklı OpenCV derlemelerinde farklı ifadeler olabilir
                gui_support = "NO"
        
        if gui_support == "NO":
            raise RuntimeError(
                "OpenCV GUI desteği bulunamadı. 'opencv-python' paketinin tam sürümünün kurulu olduğundan emin olun. "
                "'opencv-python-headless' kuruluysa, onu kaldırıp 'opencv-python' kurmayı deneyin."
            )
        else:
            print("[OpenCV Bilgisi] GUI desteği mevcut.")


        # InsightFace modelini hazırla
        print("\n[InsightFace Modeli Hazırlanıyor]")
        # Kullanılabilecek sağlayıcıları ve öncelik sırasını belirleyebilirsiniz
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        app = FaceAnalysis(allowed_modules=['detection', 'recognition']) # Sadece tespit ve tanıma modülleri
        app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 GPU, ctx_id=-1 CPU
        print("InsightFace modeli başarıyla hazırlandı (GPU kullanılıyor).")

        # Galeri yüzlerini yükle
        names, gallery_embeddings = load_gallery_embeddings(app, GALLERY_PATH)
        if not names:
            print("Galeride bilinen yüz bulunamadı. Çıkılıyor.")
            exit(1)

        # Kamera akışını başlat
        print(f"\n[Kamera Akışı Başlatılıyor: {CAMERA_ID}]")
        cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            print(f"Hata: Kamera açılamadı ({CAMERA_ID}). Lütfen kamera ID'sini veya URL'yi kontrol edin.")
            print("Olası Nedenler:")
            print("  - Dahili kamera (0) kullanılıyorsa, başka bir uygulama tarafından kullanılıyor olabilir.")
            print("  - IP kamera URL'si kullanılıyorsa:")
            print("    - URL doğru mu? (http://, rtsp://)")
            print("    - Telefonunuzdaki IP kamera uygulaması çalışıyor mu ve yayın yapıyor mu?")
            print("    - Bilgisayarınız ve telefonunuz aynı Wi-Fi ağında mı?")
            print("    - Güvenlik duvarı veya ağ ayarları bağlantıyı engelliyor mu?")
            exit(1)
        
        # Kamera çözünürlüğünü ayarlamayı deneyebilirsiniz (isteğe bağlı)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Yeniden boyutlandırılabilir pencere
        cv2.resizeWindow(WINDOW_NAME, 800, 600) # Pencere boyutunu ayarla
        print("Kamera akışı başladı... Çıkmak için 'q' tuşuna basın.")

        while True:
            ret, frame = cap.read() # Kameradan bir kare oku
            if not ret:
                print("Hata: Kameradan kare alınamadı. Akış sonlanmış olabilir.")
                # IP kameralarda bağlantı kopabilir, yeniden bağlanmayı deneyebilirsiniz (gelişmiş senaryo)
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Insightface için RGB formatına çevir
            faces = app.get(rgb_frame) # Karedeki yüzleri tespit et

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int) # Yüzün koordinatları
                probe_embedding = face.normed_embedding.reshape(1, -1) # Tespit edilen yüzün embedding'i

                # Galerideki embedding'lerle karşılaştır
                similarities = cosine_similarity(probe_embedding, gallery_embeddings)
                best_match_index = np.argmax(similarities)
                best_match_score = similarities[0, best_match_index]

                name_to_display = "Unknown"
                color = (0, 0, 255) # Bilinmeyen için kırmızı

                if best_match_score >= THRESHOLD:
                    name_to_display = names[best_match_index]
                    color = (0, 255, 0) # Tanınan için yeşil
                
                # Sonuçları kare üzerine çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name_to_display} ({best_match_score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(WINDOW_NAME, frame) # İşlenmiş kareyi göster

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # 'q' tuşuna basılırsa döngüden çık
                print("Çıkış yapılıyor...")
                break
            elif key == ord('s'): # 's' tuşuna basılırsa ekran görüntüsü al (isteğe bağlı)
                cv2.imwrite(f"screenshot_{platform.time.time()}.png", frame)
                print("Ekran görüntüsü kaydedildi!")


    except KeyboardInterrupt:
        print("\nKullanıcı tarafından kesildi.")
    except RuntimeError as e: # OpenCV GUI hatası gibi çalışma zamanı hataları için
        print(f"Çalışma Zamanı Hatası: {e}")
    except Exception as e:
        print(f"Beklenmedik Bir Hata Oluştu: {e}")
        import traceback
        traceback.print_exc() # Hatanın detaylı dökümünü yazdır
    finally:
        print("Kaynaklar serbest bırakılıyor...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Uygulama sonlandırıldı.")

