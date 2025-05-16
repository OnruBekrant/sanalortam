# base64_encoder.py
import base64

image_path = "/home/onur/projects/sanalortam/data/yusuf.jpeg" # Test resminizin yolunu buraya yazın
try:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{encoded_string}"
        print(data_url)
except FileNotFoundError:
    print(f"Hata: {image_path} bulunamadı.")