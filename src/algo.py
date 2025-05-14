
import os
try:
    import cv2
except ImportError:
    raise ImportError("cv2 modülü bulunamadı. 'pip install opencv-python' ile yükleyin.")

import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# Configuration
# =====================
GALLERY_PATH = "/home/onur/projects/sanalortam/data"  # Fotoğrafların doğrudan bulunduğu dizin
THRESHOLD = 0.5
CAMERA_ID = 0 #"https://10.38.2.254:8080/video"   #url = "http://10.80.10.62:8080/video"   # =0
#CAPTURE_BACKEND = cv2.CAP_DSHOW
WINDOW_NAME = "FaceRec"

# =====================
# Fonksiyonlar
# =====================
def load_gallery_embeddings(app, gallery_path):
    if not os.path.isdir(gallery_path):
        raise FileNotFoundError(f"Gallery path '{gallery_path}' not found.")

    names = []
    embeddings = []

    for img_name in sorted(os.listdir(gallery_path)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(gallery_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read '{img_path}'")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)
        if faces:
            name = os.path.splitext(img_name)[0]  # Dosya adından kişi ismini çıkar
            names.append(name)
            embeddings.append(faces[0].normed_embedding)
            print(f"Loaded: {name}")

    embeddings = np.vstack(embeddings) if embeddings else np.zeros((0, 512))
    return names, embeddings

# =====================
# Ana Uygulama
# =====================
if __name__ == "__main__":
    try:
        # OpenCV GUI desteği kontrolü
        build_info = cv2.getBuildInformation()
        if 'GUI:    ' in build_info and 'NO' in build_info.split('GUI:')[1].splitlines()[0]:
            raise RuntimeError("OpenCV GUI desteği bulunamadı. 'opencv-python' kurulu mu?")

        app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        names, gallery_embeddings = load_gallery_embeddings(app, GALLERY_PATH)
        if not names:
            print("No known faces found. Exiting.")
            exit(1)

        cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            print("Error: Cannot open camera.")
            exit(1)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("Streaming... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app.get(rgb_frame)
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                probe = face.normed_embedding.reshape(1, -1)
                sims = cosine_similarity(probe, gallery_embeddings)
                idx = np.argmax(sims)
                score = sims[0, idx]
                name = names[idx] if score >= THRESHOLD else "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error:", e)
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass