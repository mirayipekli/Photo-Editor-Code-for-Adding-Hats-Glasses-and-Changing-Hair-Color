# Aslıgül Kaya - 1200505017
# Okan Keskin  - 1200505044
# Miray İpekli - 1200505070

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class FaceFilterApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Filter Uygulaması")
        self.root.geometry("800x800")

        self.image = None
        self.filtered_image = None

        # Resim yükleme butonu
        self.load_button = ttk.Button(self.root, text="Fotoğraf Yükle", command=self.load_image)
        self.load_button.pack(side="top",pady=10)

        # Şapka ekleme butonu
        self.hat_button = ttk.Button(self.root, text="Gözlük ve Şapka Ekle", command=self.add_hat)
        self.hat_button.pack(side="top", padx=10)

        # Renk seçimi için ComboBox
        self.color_combo = ttk.Combobox(self.root, values=["Kırmızı", "Mor", "Yeşil", "Sarı"])
        self.color_combo.pack(side="top" ,pady=10)

        # Filtre uygula butonu
        self.apply_button = ttk.Button(self.root, text="Filtre Uygula", command=self.apply_filter)
        self.apply_button.pack(side="top" ,pady=10)

        self.save_button = ttk.Button(self.root, text="Fotoğrafı Kaydet", command=self.save_image)
        self.save_button.pack(pady=10)  

        # Resim görüntüleme alanı
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(padx= "200",fill="both", expand=True)

    def load_image(self):
        path = tk.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.image = cv2.imread(path)
            self.display_image(self.image)

    def display_image(self, image):
        # İlgili boyutlarda görüntüyü yeniden boyutlandırma
        resized_image = cv2.resize(image, (400,500))
        pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    

        # Görüntüyü Tkinter Label içinde gösterme
        image_tk = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def resize_image(self,image, width=None, height=None):
        # Görüntüyü boyutlandırma işlemlerini gerçekleştir
        if len(image.shape) > 2:
            height, curr_width, _ = image.shape
        else:
            height, curr_width = image.shape
        
        ratio = width / curr_width
        new_height = int(height * ratio)
        resized_image = cv2.resize(image, (width, new_height))
        
        return resized_image

    def create_hair_mask(self):
        height, width = self.image.shape[:2]
        # Maske görüntüsü oluştur
        hair_mask = np.zeros((height, width), dtype=np.uint8)

        # Saç maskesini oluşturmak için renk aralığını belirle
        lower_hair_color = np.array([0, 0, 0])  # Minimum renk değeri
        upper_hair_color = np.array([50, 50, 100])  # Maksimum renk değeri

        # Renk aralığına uyan pikselleri maske görüntüsüne işaretle
        hair_mask = cv2.inRange(self.image, lower_hair_color, upper_hair_color)

        cv2.imwrite('mask.jpg', hair_mask)
        print("Mask saved as mask.jpg")
        return hair_mask
    
    def change_hair_color(self,image, hair_mask, new_hair_color, opacity):
        # Renk dönüşümü için HSV formatına çevirme
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Saç maskesini uygulama
        hair_mask = cv2.cvtColor(hair_mask, cv2.COLOR_BGR2GRAY)
        _, hair_mask = cv2.threshold(hair_mask, 10, 255, cv2.THRESH_BINARY)
        
        # Saç bölgelerini seçme
        hair_pixels = np.where(hair_mask[:, :] > 0)
        
        # Saç rengini değiştirme
        if new_hair_color == 'Kırmızı':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 0   # Hue değeri (renk tonu) 0 (kırmızı) yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 1] = 255 # Saçı daha canlı göstermek için Saturation değeri (doygunluk) 255 yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 2] = opacity # Saçı daha parlak göstermek için Value değeri (parlaklık) 255 yapılır
        elif new_hair_color == 'Mor':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 150 # Hue değeri (renk tonu) 150 (mor) yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 1] = 255 # Saçı daha canlı göstermek için Saturation değeri (doygunluk) 255 yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 2] = opacity # Saçı daha parlak göstermek için Value değeri (parlaklık) 255 yapılır
        elif new_hair_color == 'Yeşil':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 60  # Hue değeri (renk tonu) 60 (yeşil) yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 1] = 255 # Saçı daha canlı göstermek için Saturation değeri (doygunluk) 255 yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 2] = opacity # Saçı daha parlak göstermek için Value değeri (parlaklık) 255 yapılır
        elif new_hair_color == 'Sarı':
            hsv_image[hair_pixels[0], hair_pixels[1], 0] = 30  # Hue değeri (renk tonu) 30 (sarı) yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 1] = 255 # Saçı daha canlı göstermek için Saturation değeri (doygunluk) 255 yapılır
            hsv_image[hair_pixels[0], hair_pixels[1], 2] = opacity # Saçı daha parlak göstermek için Value değeri (parlaklık) 255 yapılır
        
        # Renk dönüşümünü geri BGR formatına çevirme
        new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        # Saç maskesini orijinal resim boyutuna yeniden boyutlandırma
        hair_mask_resized = cv2.resize(hair_mask, (new_image.shape[1], new_image.shape[0]))
        hair_mask_resized = cv2.cvtColor(hair_mask_resized, cv2.COLOR_GRAY2BGR)
        
        # Saçı kırmızı renkteki piksellerle orijinal resim üzerine ekleme
        result = np.where(hair_mask_resized > 0, new_image, image)
        
        return result

    def add_hat(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        glass_img = cv2.imread('glasses.png', -1)
        hat_img = cv2.imread('hat.png', -1)
        if self.image is not None and self.image.any():
            # Şapka ekleme işlemi
            # Kodu tamamlayın
            # Gri tonlamaya dönüştür
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # Her yüz için gözlük ve şapka ekleyin
            for (x, y, w, h) in faces:
                # Gözlükü yeniden boyutlandırın ve konumlandırın
                resized_glasses = self.resize_image(glass_img, w)
                glass_y = int(y + h / 2) - int(resized_glasses.shape[0] / 2)
                glass_x = x
                glass_h, glass_w = resized_glasses.shape[:2]
                
                # Gözlüğü görüntüye ekleyin (alfa kanalını dikkate alarak)
                for c in range(3):
                    self.image[glass_y:glass_y+glass_h, glass_x:glass_x+glass_w, c] = (
                        resized_glasses[:, :, c] * (resized_glasses[:, :, 3] / 255.0) +
                        self.image[glass_y:glass_y+glass_h, glass_x:glass_x+glass_w, c] * (1.0 - resized_glasses[:, :, 3] / 255.0)
                    )
                
                # Şapkayı yeniden boyutlandırın ve konumlandırın
                resized_hat = self.resize_image(hat_img, width=w)
                hat_y = y - int(resized_hat.shape[0] * 0.8)
                hat_x = x
                hat_h, hat_w = resized_hat.shape[:2]
                
                # Şapka görüntüsünü ekleyin (alfa kanalını dikkate alarak)
                for c in range(3):
                    self.image[hat_y:hat_y+hat_h, hat_x:hat_x+hat_w, c] = (
                        resized_hat[:, :, c] * (resized_hat[:, :, 3] / 255.0) +
                        self.image[hat_y:hat_y+hat_h, hat_x:hat_x+hat_w, c] * (1.0 - resized_hat[:, :, 3] / 255.0)
                    )
            # Şapka eklenmiş fotoğrafı güncelleyin
            self.filtered_image = self.image.copy()
            self.display_image(self.filtered_image)

    def apply_filter(self):
        selected_color = self.color_combo.get()
        hair_mask = self.create_hair_mask()
        hair_mask = cv2.imread("mask.jpg")
        opacity = 100
        new_image = self.change_hair_color(self.image, hair_mask, selected_color, opacity)
        self.filtered_image = new_image.copy()
        self.display_image(self.filtered_image)

    def save_image(self):
        if self.filtered_image is not None and self.filtered_image.any():
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG Image", "*.jpg")])
            if save_path:
                filtered_image_rgb = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2RGB)  # Renk kanallarını düzelt
                image = Image.fromarray(filtered_image_rgb)
                image.save(save_path)
                print("Fotoğraf başarıyla kaydedildi.")
        else:
            print("Önce bir fotoğraf yükleyin ve filtreyi uygulayın.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceFilterApp()
    app.run()