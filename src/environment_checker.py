# src/environment_checker.py

import cv2
import torch
import platform # Python versiyonunu almak için

def check_environment():
    print("--- Çevre Kontrolü Başlatılıyor ---")

    # Python Sürümü
    print(f"\n[Python Bilgisi]")
    print(f"Python Sürümü: {platform.python_version()}")
    print(f"Python Yorumlayıcısı: {platform.python_implementation()} ({platform.python_compiler()})")

    # OpenCV Kontrolü
    print(f"\n[OpenCV Bilgisi]")
    try:
        print(f"OpenCV Sürümü: {cv2.__version__}")
        # Basit bir OpenCV işlemi (isteğe bağlı, GUI ortamı gerektirebilir)
        # dummy_image = cv2.imread("NON_EXISTENT_IMAGE_FOR_TEST.jpg") # Bu sadece test için
        # if dummy_image is None:
        # print("OpenCV temel fonksiyonları (dosya okuma denemesi) çalışıyor gibi görünüyor.")
        # else:
        # print("OpenCV dosya okuma denemesi beklenmedik bir sonuç verdi.")
        print("OpenCV başarıyla import edildi.")
    except ImportError:
        print("HATA: OpenCV (cv2) import edilemedi!")
    except Exception as e:
        print(f"HATA: OpenCV ile ilgili bir sorun oluştu: {e}")

    # PyTorch Kontrolü
    print(f"\n[PyTorch Bilgisi]")
    try:
        print(f"PyTorch Sürümü: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Kullanılabilir mi? : {cuda_available}")

        if cuda_available:
            print(f"  Kullanılan CUDA Sürümü (PyTorch): {torch.version.cuda}")
            print(f"  Kullanılan cuDNN Sürümü (PyTorch): {torch.backends.cudnn.version()}")
            device_count = torch.cuda.device_count()
            print(f"  Algılanan GPU Sayısı: {device_count}")
            if device_count > 0:
                for i in range(device_count):
                    print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            # Basit bir tensor işlemi
            try:
                tensor_cpu = torch.randn(2, 2)
                print(f"  CPU üzerinde oluşturulan tensor: \n{tensor_cpu}")
                tensor_gpu = tensor_cpu.to("cuda")
                print(f"  GPU'ya taşınan tensor: \n{tensor_gpu}")
                print("  PyTorch GPU işlemi başarılı.")
            except Exception as e_gpu:
                print(f"  HATA: PyTorch GPU işlemi sırasında sorun: {e_gpu}")
        else:
            print("  PyTorch için CUDA destekli GPU bulunamadı veya aktif değil.")
            print("  CPU üzerinde çalışılacak.")

    except ImportError:
        print("HATA: PyTorch import edilemedi!")
    except Exception as e:
        print(f"HATA: PyTorch ile ilgili bir sorun oluştu: {e}")

    print("\n--- Çevre Kontrolü Tamamlandı ---")

if __name__ == "__main__":
    check_environment()