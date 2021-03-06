import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array

import pytesseract as pt


import os
from PIL import Image   #----Python Imaging Library = PIL




model2 = tf.keras.models.load_model('./algila/object_detection.h5')
print('model yuklendi..')


dYolu = './foto/N230.jpeg'
foto = load_img(dYolu)   #---- PIL objesi 
foto = np.array(foto) #--- uint8: 8 bit array/unsigned int


foto1 = load_img(dYolu, target_size=(224,224))
fotoArr = img_to_array(foto1)/255.0    
#print(fotoArr)        #----------normalize cıktı

# orijinal foto boyutları
h,w,d = foto.shape
print('Foto yuksekligi =',h)  
print('Foto genisliği =',w)   

plt.figure(figsize=(10,8))
plt.imshow(foto)
plt.show()       #-------------foto gosterilir

#print(fotoArr.shape)   #----224,224,3

testArr = fotoArr.reshape(1,224,224,3)   #----yeniden yapılandırılabilir..
#print(testArr.shape)           


koordinat = model2.predict(testArr) #------okuyacagım alanın koord. 
#print("normalize koordinatlar: ", koordinat)    #---------bu veri normalize, bunu denormalize yapıcaz

denormalize = np.array([w,w,h,h])
koordinat = koordinat * denormalize
#print("gercek koordinatlar: ", koordinat)  #-------------- denormalize koor.

koordinat = koordinat.astype(np.int32)  #------float tipten int'e donusum
print("koordinatlar tam sayı degerleri: ", koordinat)    #----tam sayı koor.



###################### fotoda okunacak alan isaretleme  #########################################
xmin, xmax,ymin,ymax = koordinat[0]
ft1 =(xmin,ymin)
ft2 =(xmax,ymax)
print(ft1, ft2)   
cv2.rectangle(foto,ft1,ft2,(0,255,0),3)  #----3 pixel


plt.figure(figsize=(10,8))
plt.imshow(foto)
plt.show()     # plaka isaretlendi



foto = np.array(load_img(dYolu))
xmin ,xmax,ymin,ymax = koordinat[0]
roi = foto[ymin:ymax,xmin:xmax]  
 
plt.imshow(roi)
plt.show()

# fotodan yazıya
text = pt.image_to_string(roi)
print("okunan degeler: ",text)
