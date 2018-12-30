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


#directory = "./temp/df/df_0024318.jpg"
images_path = ".\\temp\\df\\"
train_path = ".\\data\\ISICSet\\train\\"
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


#Plot one image

#plt.imshow(image[0])
#plt.show()


'''
    Iterate and generate all image 
    variations defined in gen variable with 
    certain number of iterations
'''

#Image Data Generator Keras

gen = ImageDataGenerator(rotation_range=180, width_shift_range=0.15,
                         height_shift_range=0.1, shear_range=0.10,
                         zoom_range=0.6, channel_shift_range=35.,
                         horizontal_flip=True)


def generate_images(num_iterations):

    image = np.expand_dims(ndimage.imread(
        os.path.join(images_path, os.listdir(images_path)[3])), 0)

    aug_iter = gen.flow(image)

    for i in range(num_iterations):
        temp_image = next(aug_iter)[0].astype(np.uint8)
        aug_images.append(temp_image)


def generate_images_filters(training_dir):

    for dir in next(os.walk(training_dir))[1]:

        print("GENERATING filters for ... : ", os.path.join(training_dir, dir))

        for dirname, dirs, files in os.walk(os.path.join(training_dir, dir)):

            #filelist = get_all_images_in_dir(dir)
            #print(filelist)
            #os.chdir(".\\temp\\df")

            #for count in range(0, 2):
            for imagefile in files:
                #os.chdir("..\\df\\")
                im = Image.open(os.path.join(training_dir, dir, imagefile))
                #im = im.convert("RGB")
                #r, g, b = im.split()
                #r = r.convert("RGB")
                #g = g.convert("RGB")
                #b = b.convert("RGB")
                image_output_name_bl = os.path.splitext(imagefile)[0] + '_bl_' + os.path.splitext(imagefile)[1]
                image_output_name_us = os.path.splitext(imagefile)[0] + '_us_' + os.path.splitext(imagefile)[1]
                output_filename_bl = os.path.join(training_dir, dir, image_output_name_bl)
                output_filename_us = os.path.join(training_dir, dir, image_output_name_us)

                im_blur = im.filter(ImageFilter.GaussianBlur)
                im_unsharp = im.filter(ImageFilter.UnsharpMask)

                #os.chdir('..\\copy\\')
                #r.save(os.path.splitext(imagefile)[0] + '_r_' + str(count) + os.path.splitext(imagefile)[1])
                #g.save(os.path.splitext(imagefile)[0] + '_g_' + str(count) + os.path.splitext(imagefile)[1])
                #b.save(os.path.splitext(imagefile)[0] + '_b_' + str(count) + os.path.splitext(imagefile)[1])
                #im_blur.save(os.path.splitext(imagefile)[0] + '_bl_' + os.path.splitext(imagefile)[1])
                #im_unsharp.save(os.path.splitext(imagefile)[0] + '_us_' + os.path.splitext(imagefile)[1])

                im_blur.save(output_filename_bl)
                im_unsharp.save(output_filename_us)


#Helper function for getting all images in a directory
def get_all_images_in_dir(dir):
    filelist = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg"):
                filelist.append(file)
    return filelist

def prune_image_directory(images_dir):

    filelist = get_all_images_in_dir(images_dir)
    required_value = len(filelist) - 1500
    #prepare values for removing
    random.seed()

    print("Size before pruning: ", len(filelist))

    while len(drop_list) <= required_value:

        rnd_item = filelist[random.randint(0, len(filelist) - 1)]
        rnd_item_bl = os.path.splitext(rnd_item)[0] + '_bl_' + os.path.splitext(rnd_item)[1]
        rnd_item_us = os.path.splitext(rnd_item)[0] + '_us_' + os.path.splitext(rnd_item)[1]

        if ('us' not in rnd_item) and ('bl' not in rnd_item):

            filelist.remove(rnd_item)
            filelist.remove(rnd_item_bl)
            filelist.remove(rnd_item_us)

            if rnd_item not in drop_list:

                drop_list.append(rnd_item)
                drop_list.append(rnd_item_bl)
                drop_list.append(rnd_item_us)

        for item in drop_list:
            if os.path.isfile(os.path.join(images_dir, item)):
                os.remove(os.path.join(images_dir, item))

    print("Drop list size: ", len(drop_list))

    #Checking file list after pruning
    filelist = get_all_images_in_dir(images_dir)
    print("Size after pruning: ", len(filelist))




###############################################################
###############################################################

#Get 10 samples
#aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]


#testing Keras Image Generator
#generate_images(10)
#plots(aug_images, figsize=(40, 20), rows=2)
#plt.show()

#generate_images_filters(train_path)

copy_path=".\\temp\\nv\\"
prune_image_directory(copy_path)