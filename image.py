import matplotlib.image as mplib 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
# Reading an image and printing the shape of the image. 
img = mplib.imread('/content/opengenus_logo.png')
print(img.shape)
plt.imshow(img)

img_r = np.reshape(img, (225, 582)) 

print(img_r.shape) 

pca = PCA(32).fit(img_r) 
img_transformed = pca.transform(img_r) 
print(img_transformed.shape)
print(np.sum(pca.explained_variance_ratio_) )

temp = pca.inverse_transform(img_transformed) 
print(temp.shape)
temp = np.reshape(temp, (225,225 ,3)) 
print(temp.shape) 
plt.imshow(temp)