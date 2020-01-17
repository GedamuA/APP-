import os
import random
import time

import keras
import matplotlib
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import numpy as np
import pickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

model = keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)


def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return img, x


train_image = "/home/gede/VcGAN/Dataset/TSNE/"
Gen_image = "/home/gede/VcGAN/UESTC/Cross_view/GeneratedImage/"
img_etension = ['.png']
# max_num_images = 32

images = [os.path.join(db, f) for db, dn, filenames in os.walk(train_image) for f in filenames if
          os.path.splitext(f)[1].lower() in img_etension]
print("keeping %d images to analyze " % len(images))

tic = time.clock()
features = []
for i, image_path in enumerate(images):
    if i % 8 == 0:
        toc = time.clock()
        elap = toc - tic;
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images), elap))
        tic = time.clock()

    img, x = load_image(image_path)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

print('finished extracting features for %d images' % len(images))

features = np.array(features)
pca = PCA(n_components=15)
pca.fit(features)
pca_features = pca.transform(features)

pickle.dump([images, pca_features, pca], open('./features_caltech101.p', 'wb'))
images, pca_features, pca = pickle.load(open('./features_caltech101.p', 'rb'))

for img, f in list(zip(images, pca_features))[0:5]:
    print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... " % (img, f[0], f[1], f[2], f[3]))

num_image_plot = 100
if len(images) > num_image_plot:
    sort_order = sorted(random.sample(range(len(images)), num_image_plot))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]

x = np.array(pca_features)
print(x)
tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(x)
tx, ty = tsne[:, 0], tsne[:, 1]
tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

width = 1000
height = 1000
max_dim = 75
import PIL

full_image = PIL.Image.new('RGBA', (width, height))
for img, x, y in zip(images, tx, ty):
    tile = PIL.Image.open(img)
    rs = max(1, tile.width / max_dim, tile.height / max_dim)
    tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), PIL.Image.ANTIALIAS)
    full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))
import matplotlib as plt
import matplotlib.pyplot as plt

matplotlib.pyplot.figure(figsize=(16, 12))
plt.imshow(full_image)
matplotlib.pyplot.show()
full_image.save("image1")
