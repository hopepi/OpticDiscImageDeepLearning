import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from ImageSharpened import Sharpened
from ImageTrashContur import OpticDiscTrash
from SelectDirectory import SelectDirectory
from ImageColorClahe import ColorCLAHE
from ImageManuelZoom import ImageSelector
from ModelTest import ImageModelPredictor
import warnings
warnings.filterwarnings("ignore")


global current_image

def main_screen():
    global window,predict_button,image_label
    window = tk.Tk()
    window.geometry("800x600")
    window.title("Glaucoma Hastalık Teşhisi")

    window.protocol("WM_DELETE_WINDOW", window.quit)

    info_label = tk.Label(window, text="Bu uygulama, glokom teşhisi için göz görüntülerini analiz etmektedir. "
                                       "Lütfen bir resim seçin ve tahmin yapmak için butona tıklayın. "
                                       "Otomatik teşhis başarılı olabilmesi için, optik diskten yaklaşık 100 piksel "
                                       "çevresinin kapsandığından emin olun.",
                          wraplength=700)
    info_label.pack(pady=10)

    dosya_yolu_label = tk.Label(window, text="Seçilen dosya yok", wraplength=700)
    dosya_yolu_label.pack(pady=10)

    dosya_yolu_buton = tk.Button(window, text="Resminizi Seçiniz",
                                 command=lambda: select_image(dosya_yolu_label, window))
    dosya_yolu_buton.pack(pady=10)

    image_label = tk.Label(window)
    image_label.pack(pady=10)

    predict_button = tk.Button(window, text="Tahmin Yap", command=predict_all_model)
    predict_button.pack(pady=10)
    predict_button.pack_forget()

    window.mainloop()



def apply_manuel_zoom(resim, window):
    global image_label,current_image
    selector = ImageSelector(resim)
    selected_region = selector.select_region()

    if selected_region is not None:

        clahe = cv2.createCLAHE(clipLimit=3.0,
                                tileGridSize=(8, 8))  # Daha sert uygulama için clipLimit ve tileGridSize artırıldı

        # Renkli görüntüde CLAHE uygulama
        lab_image = cv2.cvtColor(selected_region, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab_image)
        l_channel_clahe = clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_channel_clahe, a, b))

        current_image = lab_clahe #lab_clahe
        selected_pil_img = Image.fromarray(cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)).resize((224, 224), Image.LANCZOS)
        photo = ImageTk.PhotoImage(selected_pil_img)

        # Resmi değiştirme
        image_label.config(image=photo)
        image_label.image = photo

        cv2.imshow("Selected Region", selected_region)
        predict_button.pack()
    else:
        messagebox.showinfo("Uyarı", "Seçim yapılmadı. Lütfen bir alan seçin.")


def select_image(label, window):
    global image_label,current_image

    test = SelectDirectory()
    resim_yolu = test.resim_dosya_yolu()

    if resim_yolu:
        label.config(text=resim_yolu)
        final_img = apply_clahe_and_sharpen_and_trash(resim_yolu)
        cv2_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img_rgb).resize((224, 224), Image.LANCZOS)

        if final_img is None:
            messagebox.showwarning("Uyarı", "İşleme sonucu uygun değil. Manuel seçim yapmanız gerekiyor.")
            manuel_zoom_pic = apply_clahe_and_sharpen(resim_yol=resim_yolu)
            apply_manuel_zoom(manuel_zoom_pic, window)
        else:
            photo = ImageTk.PhotoImage(pil_img)
            current_image = final_img

            image_label.config(image=photo)
            image_label.image = photo

            if messagebox.askyesno("Sonuç", "Sonuçtan memnun musunuz?"):
                predict_button.pack()
            else:
                manuel_zoom_pic = apply_clahe_and_sharpen(resim_yol=resim_yolu)
                apply_manuel_zoom(manuel_zoom_pic, window)




def apply_clahe_and_sharpen(resim_yol):
    color_clahe = ColorCLAHE()
    clahe_img = color_clahe.apply_clahe_path(resim_yol)

    sharpened = Sharpened(clahe_img)
    sharpened.resim = clahe_img
    sharpened_img = sharpened.sharpened_path()

    return sharpened_img




def apply_optic_disc_trash(sharpened_img):
    optic_disc_trash = OpticDiscTrash(resim_yol=None)
    optic_disc_trash.resim = sharpened_img
    final_image = optic_disc_trash.apply_clahe()

    return final_image




def apply_clahe_and_sharpen_and_trash(resim_yol):
    sharpened_img = apply_clahe_and_sharpen(resim_yol)

    if sharpened_img is not None:
        final_image = apply_optic_disc_trash(sharpened_img)

        if final_image is not None:

            return final_image

    return None


def predict_all_model():
    global current_image
    if current_image is not None:
        model_path_googleNet = "GoogleNet.h5"
        model_path_resNet101 = "ResNet101.h5"
        model_path_mobileNetV2 = "MobileNetV2.h5"

        model_predictor_resNet101 = ImageModelPredictor(model_path_resNet101)
        prediction_resNet101 = model_predictor_resNet101.get_prediction(current_image)

        model_predictor_mobileNetV2 = ImageModelPredictor(model_path_mobileNetV2)
        prediction_mobileNetV2 = model_predictor_mobileNetV2.get_prediction(current_image)

        model_predictor_googleNet = ImageModelPredictor(model_path_googleNet)
        prediction_googleNet = model_predictor_googleNet.get_prediction(current_image)

        messagebox.showinfo("Tahminler", f"Tahmin sonucu GoogleNet Glaucoma İhtimali : {prediction_googleNet[0][0]}\n"
                                         f"Tahmin sonucu GoogleNet Glaucoma olmama İhtimali : {prediction_googleNet[0][1]}")
        messagebox.showinfo("Tahminler", f"Tahmin sonucu ResNet101  Glaucoma İhtimali : {prediction_resNet101[0][0]}\n"
                                         f"Tahmin sonucu ResNet101 Glaucoma olmama İhtimali : {prediction_resNet101[0][1]}")
        messagebox.showinfo("Tahminler", f"Tahmin sonucu MobileNetV2  Glaucoma İhtimali : {prediction_mobileNetV2[0][0]}\n"
                                         f"Tahmin sonucu MobileNetV2 Glaucoma olmama İhtimali : {prediction_mobileNetV2[0][1]}")
    else:
        messagebox.showwarning("Uyarı", "Resim bulunamadı.")

main_screen()
