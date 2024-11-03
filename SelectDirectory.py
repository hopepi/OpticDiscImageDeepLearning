import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
class SelectDirectory:
    def __init__(self):
        pass

    def resim_dosya_yolu(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Lütfen resmi seçiniz",
            filetypes=[("Image files", "*png;*jpg;*jpeg")]
        )

        if file_path:
            return file_path
        else:
            messagebox.showerror("Lütfen düzgün resim seçiniz")
            return None