import glob
import os
import random
import string
import tensorflow as tf
import keras
import scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage

directory = "./temp/df/df_0024318.jpg"
image_path = "./temp/df/"
aug_images = []
drop_list=[]

'''
Plots generated images with Image generator
TODO check
'''
def plots(ims, figsize=(12,6), rows = 1, interp = False, titles = None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize = 16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.15,
                         height_shift_range=0.1, shear_range=0.20,
                         zoom_range=0.5, channel_shift_range=35.,
                         horizontal_flip=True)

#Plot one image
image = np.expand_dims(ndimage.imread(directory),0)
plt.imshow(image[0])
plt.show()

'''
    Iterate and generate all image 
    variations defined in gen variable with 
    certain number of iterations
'''
aug_iter = gen.flow(image)

def generate_images(num_iterations):
    for i in range(num_iterations):
        temp_image = next(aug_iter)[0].astype(np.uint8)
        aug_images.append(temp_image)


def get_all_images_in_dir(dir):
    filelist = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg"):
                filelist.append(file)
    return filelist

def generate_images_filters(image_path):

    filelist = get_all_images_in_dir(image_path)

    print(filelist)
    os.chdir("./temp/df")
    for count in range(0, 2):
        for imagefile in filelist:
            os.chdir("../df/")
            im = Image.open(imagefile)
            im = im.convert("RGB")
            r, g, b = im.split()
            r = r.convert("RGB")
            g = g.convert("RGB")
            b = b.convert("RGB")
            im_blur = im.filter(ImageFilter.GaussianBlur)
            im_unsharp = im.filter(ImageFilter.UnsharpMask)

            os.chdir('../copy/')
            r.save(os.path.splitext(imagefile)[0] + '_r_' + str(count) + os.path.splitext(imagefile)[1])
            g.save(os.path.splitext(imagefile)[0] + '_g_' + str(count) + os.path.splitext(imagefile)[1])
            b.save(os.path.splitext(imagefile)[0] + '_b_' + str(count) + os.path.splitext(imagefile)[1])
            im_blur.save(os.path.splitext(imagefile)[0] + '_bl_' + str(count) + os.path.splitext(imagefile)[1])
            im_unsharp.save(os.path.splitext(imagefile)[0] + '_us_' + str(count) + os.path.splitext(imagefile)[1])


def prune_image_directory(image_dir):

    filelist = get_all_images_in_dir(image_dir)
    print(len(filelist))

    #prepare values for removing
    random.seed()
    for i in range(len(filelist)):
        if i % 2 == 0:
            rnd_item = filelist[random.randint(0, len(filelist) - 1)]
            filelist.remove(rnd_item)
            if rnd_item not in drop_list:
                drop_list.append(rnd_item)

    print(len(drop_list))

    for item in drop_list:
        exists = os.path.isfile(os.path.join(image_dir, item))
        if exists:
            os.remove(os.path.join("temp/copy/", item))
        else:
            print("Error - file not found!")


    filelist = get_all_images_in_dir(image_dir)
    print(len(filelist))





###############################################################
###############################################################

#Get 10 samples
#aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(56)]

'''
try:
    generate_images(36)
    plots(aug_images, figsize=(40, 20), rows=4)
    plt.show()
except:
    print("Error occured: " , OSError)
'''


copy_path="./temp/copy/"
prune_image_directory(copy_path)