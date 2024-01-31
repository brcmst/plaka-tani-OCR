# License Plate Detection and Text Recognition

Bu proje, bir görüntüdeki plaka koordinatlarını tespit eden bir nesne tanıma modelini ve ardından plaka bölgesini kırpıp optik karakter tanıma (OCR) kullanarak plakadaki metni çıkaran bir yazılımı içermektedir.

## Kullanılan Teknolojiler

- TensorFlow: Nesne tanıma modelinin oluşturulması ve eğitimi için kullanılmıştır.
- OpenCV: Görüntü işleme işlemleri için kullanılmıştır.
- Matplotlib: Sonuçların görselleştirilmesi için kullanılmıştır.
- pytesseract: OCR işlemi için kullanılmıştır.
- PIL (Python Imaging Library): Görüntü işleme işlemleri için kullanılmıştır.

## Model Yükleme

TensorFlow ile eğitilmiş bir nesne tanıma modeli (`object_detection.h5`) kullanılmıştır. Bu model, plakanın koordinatlarını tahmin edebilmektedir.

```python
model2 = tf.keras.models.load_model('./algila/object_detection.h5')


