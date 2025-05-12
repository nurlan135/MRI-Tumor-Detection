import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from tensorflow.keras import layers
# Transfer Learning üçün əsas model və ön emal funksiyası
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
# Sinif ağırlıqları üçün import hələ də qalır, amma istifadə etməyəcəyik
from sklearn.utils.class_weight import compute_class_weight

# --- Konfiqurasiya ---
base_dir = pathlib.Path('Dataset/')
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'

if not train_dir.exists() or not test_dir.exists():
    raise FileNotFoundError(f"Təlim ({train_dir}) və ya Test ({test_dir}) qovluğu tapılmadı. ")

# Şəkil və Təlim Parametrləri
img_height = 128
img_width = 128
# Əvvəlki uğurlu təlimdə istifadə etdiyimiz batch size-a qayıdaq (və ya saxlayaq)
batch_size = 8 # Yaddaş xətası verməyən son batch ölçüsü
epochs = 15 # Yalnız ilkin təlim epoxa sayı

print(f"Təlim qovluğu: {train_dir}")
print(f"Test/Validasiya qovluğu: {test_dir}")
print(f"Batch ölçüsü: {batch_size}")

# --- Məlumatların Yüklənməsi ---
print("Təlim məlumatları yüklənir...")
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir, seed=123, image_size=(img_height, img_width),
  batch_size=batch_size, label_mode='binary'
)
print("Test/Validasiya məlumatları yüklənir...")
val_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir, seed=123, image_size=(img_height, img_width),
  batch_size=batch_size, label_mode='binary', shuffle=False
)
class_names = train_ds.class_names
print("Tapılan siniflər:", class_names)

# Dataset ölçülərini hesablamaq
try:
    num_train_samples = sum(1 for _ in train_dir.glob('*/*.*'))
    num_val_samples = sum(1 for _ in test_dir.glob('*/*.*'))
    print(f"Təlim şəkillərinin sayı (tapılan fayllar): {num_train_samples}")
    print(f"Test/validasiya şəkillərinin sayı (tapılan fayllar): {num_val_samples}")
except Exception as e:
    print(f"Fayl sayını hesablayarkən xəta: {e}.")

# --- Ön Emal (Preprocessing) ---
AUTOTUNE = tf.data.AUTOTUNE
def preprocess_data(image, label):
  image = tf.cast(image, tf.float32)
  image = preprocess_input(image)
  return image, label
print("\nDatasetlərə MobileNetV2 üçün ön emal tətbiq edilir...")
train_ds = train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(max(10, int(num_train_samples/4))).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print("Ön emal və optimallaşdırma tamamlandı.")

# --- Validasiya Seti Etiket Yoxlaması ---
print("\nValidasiya (Test) setindəki etiketlərin sayı yoxlanılır...")
val_labels_list = []
try:
    for _, labels_batch in val_ds.as_numpy_iterator():
        val_labels_list.extend(labels_batch)
    if len(val_labels_list) > 0:
        val_labels_array = np.array(val_labels_list)
        unique_val, counts_val = np.unique(val_labels_array, return_counts=True)
        val_label_counts = dict(zip(unique_val, counts_val))
        print(f"Validasiya setindəki etiketlərin sayı: {val_label_counts} (0: {class_names[0]}, 1: {class_names[1]})")
        if 0 not in val_label_counts or val_label_counts.get(0, 0) == 0:
             print("XƏBƏRDARLIQ: Validasiya setində sağlam (etiket 0) nümunə tapılmadı!")
        else:
            print(f"Validasiya setində {val_label_counts.get(0, 0)} ədəd sağlam (etiket 0) nümunə var.")
    else:
        print("XƏBƏRDARLIQ: Validasiya setindən etiketlər oxuna bilmədi.")
except Exception as e:
    print(f"Validasiya etiketlərini yoxlayarkən xəta: {e}")

# --- Transfer Learning Modelinin Qurulması ---
print("\nTransfer Learning modeli (MobileNetV2 əsaslı) qurulur...")
input_shape = (img_height, img_width, 3)
base_model = MobileNetV2(input_shape=input_shape,
                           include_top=False,
                           weights='imagenet')
base_model.trainable = False # Dondurulmuş qalır
print(f"MobileNetV2 əsas modelinin çəkiləri donduruldu.")
inputs = keras.Input(shape=input_shape, name="input_layer")
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dropout(0.3, name="dropout_1")(x)
outputs = layers.Dense(1, activation='sigmoid', name="output_layer")(x)
model = Model(inputs, outputs, name="MobileNetV2_Transfer")
print("Transfer Learning modeli quruldu.")

# --- Modelin Kompilyasiyası ---
print("\nModel təlim üçün kompilyasiya edilir...")
initial_lr = 0.0005
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')
                      ])
print("Model kompilyasiya edildi.")

# --- Modelin Təlimi ---
print(f"\nModel təlimə başlayır ({epochs} epoxa)...")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs # Yalnız ilkin epoxa sayı
)
print("Modelin təlimi başa çatdı.")

# --- Fine-tuning Mərhələsi Ləğv Edildi ---

# --- Modelin Qiymətləndirilməsi (İlkin Təlim Sonrası) ---
print("\nModel (İlkin təlim sonrası) test seti üzərində yekun qiymətləndirilir...")
results = model.evaluate(val_ds, verbose=1)
print("\nTest Seti Üzrə Nəticələr:")
metric_dict = {}
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")
    metric_dict[name] = value

print("\nQarışıqlıq Matrisi və Proqnoz Ehtimalları hesablanır...")
y_pred_prob = model.predict(val_ds)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten() # 0.5 həddi ilə təsnifat

print("Əsl validasiya etiketləri təkrar əldə edilir...")
y_true_list = []
try:
    for _, labels_batch in val_ds.as_numpy_iterator():
        y_true_list.extend(labels_batch)
    y_true = np.array(y_true_list)
    print(f"{len(y_true)} ədəd əsl etiket əldə edildi.")

    if len(y_true) != len(y_pred_prob):
         print(f"XƏBƏRDARLIQ: Əsl etiket sayı ({len(y_true)}) proqnoz sayı ({len(y_pred_prob)}) ilə uyğun gəlmir!")
         min_len = min(len(y_true), len(y_pred_prob))
         y_true = y_true[:min_len]
         y_pred_prob = y_pred_prob[:min_len]
         y_pred = y_pred[:min_len]
         print(f"Hər iki massiv {min_len} uzunluğuna qısaldıldı.")

    # --- Sağlam Nümunələr Üçün Ehtimal Analizi ---
    print("\nSağlam şəkillər (əsl etiket=0) üçün proqnoz ehtimalları:")
    healthy_indices = np.where(y_true == 0)[0]
    if len(healthy_indices) > 0:
        print(f"Tapılan sağlam nümunə sayı: {len(healthy_indices)}")
        healthy_probs = y_pred_prob[healthy_indices].flatten()
        print(f"  Minimum ehtimal: {np.min(healthy_probs):.4f}")
        print(f"  Maksimum ehtimal: {np.max(healthy_probs):.4f}")
        print(f"  Orta ehtimal: {np.mean(healthy_probs):.4f}")
        print(f"  Median ehtimal: {np.median(healthy_probs):.4f}")
        above_threshold = np.sum(healthy_probs > 0.5)
        print(f"  0.5 həddindən yuxarı olanların sayı: {above_threshold}")

        print("  İlk 15 sağlam nümunə üçün proqnoz ehtimalları:")
        for i in range(min(15, len(healthy_indices))):
            idx = healthy_indices[i]
            print(f"    Əsl: {y_true[idx]}, Ehtimal: {y_pred_prob[idx][0]:.4f}, Proqnoz (0.5 həddi): {y_pred[idx]}")
    else:
        print("  Validasiya setində sağlam nümunə tapılmadı.")

    # Qarışıqlıq matrisini yarat
    cm = confusion_matrix(y_true, y_pred)

    # --- Matrisin Çəkilməsi və Detallar (Try blokunun içində) ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure() # Yeni fiqur yaradırıq
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca()) # Mövcud fiqurun oxlarına çəkirik
    plt.title("Qarışıqlıq Matrisi (Test/Validasiya Seti - İlkin Təlim)")
    plt.savefig('confusion_matrix_initial_train.png') # Fayla yazırıq
    plt.close() # Fiquru bağlayırıq ki, göstərilməsin (plt.show() əvəzinə)

    print("\nQarışıqlıq Matrisinin detalları:")
    if len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"  Doğru Neqativ (TN): {tn} ({class_names[0]} -> {class_names[0]})")
        print(f"  Yanlış Pozitiv (FP): {fp} ({class_names[0]} -> {class_names[1]})")
        print(f"  Yanlış Neqativ (FN): {fn} ({class_names[1]} -> {class_names[0]})")
        print(f"  Doğru Pozitiv (TP): {tp} ({class_names[1]} -> {class_names[1]})")

        total = tn + fp + fn + tp
        if total > 0:
            accuracy_cm = (tp + tn) / total
            precision_cm = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_cm = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity_cm = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"\nMatrisdən hesablanan metrikalar (təxmini):")
            print(f"  Accuracy: {accuracy_cm:.4f}")
            print(f"  Precision (Tumor): {precision_cm:.4f}")
            print(f"  Recall (Tumor): {recall_cm:.4f}")
            print(f"  Specificity (Healthy): {specificity_cm:.4f}")
        else:
            print("\nMatris boşdur.")
    else:
        print("\nQarışıqlıq matrisi 2x2 formatında deyil.")
        print(cm)
    # --- Matris Çəkmə və Detallar Blokunun Sonu ---

except Exception as e:
    print(f"XƏTA: Nəticələr analiz edilərkən xəta baş verdi: {e}")


# --- Təlim Tarixçəsini Vizuallaşdırmaq (Yalnız İlkin Təlim) ---
print("\nTəlim tarixçəsi (yalnız ilkin təlim) vizuallaşdırılır...")
available_metrics = list(history.history.keys())
print(f"Tarixçədə mövcud metrikalar: {available_metrics}")

plt.figure(figsize=(12, 8)) # Yeni fiqur
total_epochs_trained = len(history.history['loss'])
epochs_range = range(total_epochs_trained)
plot_index = 1

# Qrafikləri çəkək
if 'accuracy' in available_metrics and 'val_accuracy' in available_metrics:
    plt.subplot(2, 2, plot_index); plt.plot(epochs_range, history.history['accuracy'], label='Təlim Accuracy'); plt.plot(epochs_range, history.history['val_accuracy'], label='Validasiya Accuracy'); plt.title('Dəqiqlik (Accuracy)'); plt.xlabel(f'Epoxa'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True); plot_index += 1
if 'loss' in available_metrics and 'val_loss' in available_metrics:
    plt.subplot(2, 2, plot_index); plt.plot(epochs_range, history.history['loss'], label='Təlim Loss'); plt.plot(epochs_range, history.history['val_loss'], label='Validasiya Loss'); plt.title('İtki (Loss)'); plt.xlabel(f'Epoxa'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plot_index += 1
if 'precision' in available_metrics and 'val_precision' in available_metrics:
    plt.subplot(2, 2, plot_index); plt.plot(epochs_range, history.history['precision'], label='Təlim Precision'); plt.plot(epochs_range, history.history['val_precision'], label='Validasiya Precision'); plt.title('Precision'); plt.xlabel(f'Epoxa'); plt.ylabel('Precision'); plt.legend(); plt.grid(True); plot_index += 1
if 'recall' in available_metrics and 'val_recall' in available_metrics:
    plt.subplot(2, 2, plot_index); plt.plot(epochs_range, history.history['recall'], label='Təlim Recall'); plt.plot(epochs_range, history.history['val_recall'], label='Validasiya Recall'); plt.title('Recall (Sensitivity)'); plt.xlabel(f'Epoxa'); plt.ylabel('Recall'); plt.legend(); plt.grid(True);

plt.tight_layout()
plt.savefig('training_history_initial_train.png') # Fayla yazırıq
plt.close('all') # Bütün açıq Matplotlib fiqurlarını bağlayırıq

# Modeli saxlamaq
print("\nModel yaddaşa verilir...")
model.save('mri_tumor_detector_mobilenetv2_initial.h5') # Fayl adı
print("Model 'mri_tumor_detector_mobilenetv2_initial.h5' faylında yaddaşa verildi.")

print("\nProses başa çatdı.")
