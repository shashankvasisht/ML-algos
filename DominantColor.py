
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import cv2
from sklearn.cluster import KMeans


# In[43]:


im = cv2.imread('./im.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

out_r = 100
im = cv2.resize(im, (int(out_r*float(c)/r), out_r))
print im.shape

pixels = im.reshape((-1, 3))
print pixels.shape

plt.imshow(im)
plt.show()


# In[17]:


km = KMeans(n_clusters=8)
km.fit(pixels)


# In[23]:


centr_colors = np.array(km.cluster_centers_, dtype='uint8')
print centr_colors.dtype
print centr_colors

print centr_colors.shape


# In[29]:


freq = np.array(np.unique(km.labels_, return_counts=True)[1], dtype='float32')
print freq
print freq.sum()
freq = freq/pixels.shape[0]
print freq


# In[39]:


dom = [[freq[ix], centr_colors[ix]] for ix in range(km.n_clusters)]

DOM = sorted(dom, key=lambda z:z[0], reverse=True)
#DOM = np.array(DOM)
print DOM[0][1]
#print DOM.shape


# In[31]:


for ix in DOM:
    print ix
    print "----------"


# In[41]:


patch = np.ones((50, 500, 3))
start = 0
for ix in range(km.n_clusters):
    width = int(DOM[ix][0]*patch.shape[1])
    end = start+width
    patch[:,start:end,:] = 255 - DOM[ix][1]
    start = end
    plt.axis("off")
    plt.imshow(patch)
plt.show()


# In[42]:


patch = np.ones((50, 500, 3))
start = 0
for ix in range(km.n_clusters):
    width = int(DOM[ix][0]*patch.shape[1])
    end = start+width
    patch[:,start:end,:] = DOM[ix][1]
    start = end
    plt.axis("off")
    plt.imshow(patch)
plt.show()

