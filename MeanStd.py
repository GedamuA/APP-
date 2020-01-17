from PIL import Image
from numpy import asarray

img1 = "/home/gede/VcGAN/Dataset/Train_image/a00_v0_d1_p001_c1.png"
img2 = "/home/gede/VcGAN/Dataset/Test_image/a00_v1_d1_p002_c2.png"
img3 = "/home/gede/VcGAN/UESTC/Cross_view/GeneratedImage/a00_ov0_gv1_d1_p002_c1.png"


def open(path):
    image = Image.open(path)
    print(image.format)
    print(image.mode)
    print(image.size)

    #image.show()


def imge_asarray(image):
    image = Image.open(image)
    pixel = asarray(image)
    print("Data type : %s" % pixel.dtype)
    print("min: %.3f, max is:  %.3f" % (pixel.min(), pixel.max()))
    print(pixel.mean())
    pixel = pixel.astype(float)
    pixel = pixel/255
    print("min: %.3f, max is:  %.3f" % (pixel.min(), pixel.max()))
    print(pixel.mean())

#
#
# imge_asarray(img1)
# imge_asarray(img2)
imge_asarray(img3)
