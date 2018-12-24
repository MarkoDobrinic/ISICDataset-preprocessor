import csv
import glob
import os
import pathlib
import re
import shutil
from shutil import copyfile
import numpy as np
import pandas as pd



###########################################################################################

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

###########################################################################################

current_folder = os.getcwd()
class_names = []
colnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
images_folder = "D:\DIPLOMSKI\Dataset\ISIC2018_Task3_Training_Input"
list_MEL, list_NV, list_BCC, list_AKIEC, list_BKL, list_DF, list_VASC = ([] for i in range(7))

#sanity checking no. of images
def count_total_images(images_folder):
    counter = 0
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
                #print(os.path.join(directory, filename))
                counter += 1
    print("num files: ", counter)

'''
    Create a list of categories according to ISIC ground truth .csv file
'''
def create_class_name_list(colnames_list):
    for list in colnames_list:
        if list != "image":
            class_names.append(list.lower())
    print(class_names)

'''
    Import ISIC .csv file into pandas and extract image names according to
    one-hot-encoded labels in csv into separate lists for each category
'''
def pandas_reader_csv(file_obj):

    csv_data = pd.read_csv(csv_path, names=colnames, delimiter=',')

    for index, row in csv_data.iterrows():
        if row[0] == str("image"):
            continue
        if row[1] != str(0.0):
            list_MEL.append(row[0])
        elif row[2] != str(0.0):
            list_NV.append(row[0])
        elif row[3] != str(0.0):
            list_BCC.append(row[0])
        elif row[4] != str(0.0):
            list_AKIEC.append(row[0])
        elif row[5] != str(0.0):
            list_BKL.append(row[0])
        elif row[6] != str(0.0):
            list_DF.append(row[0])
        else:
            list_VASC.append(row[0])

    print("Images of MEL: ", len(list_MEL))
    print("Images of NV: ", len(list_NV))
    print("Images of BCC: ", len(list_BCC))
    print("Images of AKIEC: ", len(list_AKIEC))
    print("Images of BKL: ", len(list_BKL))
    print("Images of DF: ", len(list_DF))
    print("Images of VASC: ", len(list_VASC))

    print("Total images: ", len(list_MEL)+len(list_NV)+ len(list_BCC)
          + len(list_AKIEC) + len(list_BKL) + len(list_DF) + len(list_VASC))


'''
Create directories for training and test
if don't exists for each category from ISIC dataset.
'''
def create_directories():
    # Create a list of dirs for each class, e.g.:
    # ['knifey-spoony/test/forky/',
    #  'knifey-spoony/test/knifey/',
    #  'knifey-spoony/test/spoony/']

    for ISIC_class in class_names:
        if not os.path.exists("data/ISICset/{}/{}".format("train", ISIC_class)):
            pathlib.Path('data/ISICset/{}/{}'.format("train", ISIC_class)).mkdir(parents=True, exist_ok=True)

        if not os.path.exists("data/ISICset/{}/{}".format("test", ISIC_class)):
            pathlib.Path('data/ISICset/{}/{}'.format("test", ISIC_class)).mkdir(parents=True, exist_ok=True)


'''
Copy actual images from ISIC dataset according to categories to
their own training directories
'''
def find_and_copy_images_to_training(current_list, folder_ext):
    print("Copying Image files to {} folder ... ".format(folder_ext))

    dst_dir = "data/ISICset/train/{}/".format(folder_ext)

    for item in current_list:
        for file in glob.glob(images_folder + "\\" + item + '*', recursive=True):
            if file not in glob.glob(dst_dir + "\\" + item + '*', recursive=True):
                shutil.copy(file, dst_dir)
                dst_file = os.path.join(dst_dir, file)
                new_dst_file_name = os.path.join(dst_dir, file)
                os.rename(dst_file, new_dst_file_name)

    print("Files copied >> {}".format(folder_ext))
    return



def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename,
                os.path.join(dir, titlePattern % title + ext))



def replace(dir):
    name_map = {
        "ISIC": ""
    }
    for root, dirs, files in os.walk(dir):
        for f in files:
            for name in name_map.keys():
                if re.search(name, f) != None:
                    new_name = re.sub(name, name_map[name], f)
                    try:
                        os.rename(os.path.join(root, f), os.path.join(root, new_name))
                    except OSError:
                        print("No such file or directory!")

'''
    For each subdirectory in train images folder,
    take the name of the subdirectory and according to that name
    rename all the files inside and replace (remove) ISIC part of the name        
'''
def replace_and_rename(training_dir):
    for dir in next(os.walk(training_dir))[1]:
        rename(r'./data/ISICset/train/' + "\\" + dir, r'*.jpg', r'{}%s'.format(dir))
        replace('./data/ISICset/train/' + "\\" + dir)


def total_images_copied(training_dir):
    total = 0
    for root, dirs, files in os.walk(training_dir):
        total += len(files)

    print("Total images copied across categories :", total)





if __name__ == "__main__":
    csv_path = "D:\DIPLOMSKI\Dataset\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"
    with open(csv_path, "rt") as f_obj:
        pandas_reader_csv(f_obj)
        create_directories()
        count_total_images(images_folder)

        '''
            Copy all images to data directory
        '''
        #find_and_copy_images_to_training(list_VASC, "vasc")
        #find_and_copy_images_to_training(list_NV, "nv")
        #find_and_copy_images_to_training(list_MEL, "mel")
        #find_and_copy_images_to_training(list_DF, "df")
        #find_and_copy_images_to_training(list_BKL, "bkl")
        #find_and_copy_images_to_training(list_BCC, "bcc")
        #find_and_copy_images_to_training(list_AKIEC, "akiec")


        #rename(r"./data/ISICset/train/df/", r'*_*.jpg', r'df%s')
        #replace(r"./data/ISICset/train/df/")

        #replace_and_rename('./data/ISICset/train/')
