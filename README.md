# Beyin MRİ Şiş Təsnifatı Modeli

Bu layihə, beyin MRİ görüntülərindən şişin olub-olmamasını təsnif etmək üçün TensorFlow/Keras və Transfer Learning (MobileNetV2) istifadə edərək yaradılmış bir Konvolyusiya Neyron Şəbəkəsi (CNN) modelini ehtiva edir.

## Layihə Haqqında

Bu model, sağlam və şişli beyin MRİ təsvirlərindən ibarət bir dataset üzərində təlim edilmişdir. İlkin təlim mərhələsində MobileNetV2 modelinin dondurulmuş çəkiləri istifadə edilmiş, daha sonra təsnifat başlığı təlim edilmişdir. (Əgər fine-tuning uğurlu olsaydı, onu da qeyd edərdiniz).

## Nəticələr (İlkin Təlim Sonrası)

Əldə edilən ən yaxşı model (yalnız ilkin təlim, batch_size=8) test/validasiya seti üzərində aşağıdakı performansı göstərdi:

*   **Accuracy:** ~97.44%
*   **Precision (Tumor):** ~96.75%
*   **Recall (Tumor):** ~98.03%
*   **Specificity (Healthy):** ~96.88%

**Qarışıqlıq Matrisi:**
*   TN: 155 | FP: 5
*   FN: 3   | TP: 149

(Buraya `confusion_matrix_initial_train.png` şəklini də əlavə edə bilərsiniz, əgər GitHub-a yükləyirsinizsə).

## Fayl Strukturu

*   `train_model.py`: Modeli təlim etmək üçün Python skripti.
*   `mri_tumor_detector_mobilenetv2_initial.h5`: Əvvəlcədən təlim edilmiş Keras modeli faylı.
*   `Dataset/`: (Əgər yükləyirsinizsə) Təlim və test üçün istifadə olunan şəkilləri ehtiva edən qovluq (`train` və `test` alt qovluqları ilə).
*   `README.md`: Bu fayl.
*   `requirements.txt`: Lazımi Python kitabxanaları.
*   `.gitignore`: Git tərəfindən izlənilməyəcək fayllar.

## Quraşdırma və İstifadə

1.  **Klondanma:** Bu repozitoriyanı klonlayın:
    ```bash
    git clone <repo_url>
    cd <repo_folder>
    ```
2.  **Kitabxanaları Quraşdırma:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dataset:** Əgər dataset repozitoriyada yoxdursa, onu əldə edib `Dataset/train/` və `Dataset/test/` qovluqlarına yerləşdirin.
4.  **Təlim (Əgər təkrar təlim etmək istəsəniz):**
    ```bash
    python train_model.py
    ```
    Bu, yeni bir `.h5` model faylı yaradacaq.
5.  **Modeldən İstifadə:** Saxlanılmış `.h5` faylını yükləyərək yeni proqnozlar üçün istifadə edə bilərsiniz (bunun üçün ayrıca bir skript və ya Jupyter Notebook yaratmaq faydalı olardı).

## Asılılıqlar

Əsas asılılıqlar `requirements.txt` faylında göstərilib (TensorFlow, Matplotlib, Scikit-learn, NumPy, Pillow).