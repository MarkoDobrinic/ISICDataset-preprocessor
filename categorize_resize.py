import glob
import os
import pathlib
import random
import re
import shutil
from os.path import dirname, join
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from image_processor import get_all_images_in_dir

'''
    Loading .env file - environment variables
'''
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


'''
    Setting local variables
'''
current_folder = os.getcwd()
class_names = []
colnames = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
IMAGES_FOLDER = os.environ.get("IMAGES_TRAINING_DIR")
list_MEL, list_NV, list_BCC, list_AKIEC, list_BKL, list_DF, list_VASC = ([] for i in range(7))


'''
    Create a list of categories according to ISIC ground truth .csv file
'''
for list in colnames:
    if list != "image":
        class_names.append(list.lower())

#sanity checking no. of images
def count_total_images(images_folder):
    counter = 0
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
                #print(os.path.join(directory, filename))
                counter += 1
    print("\nTotal number of files in original dataset: ", counter)


'''
    Import ISIC .csv file into pandas and extract image names according to
    one-hot-encoded labels in csv into separate lists for each category
'''
def pandas_reader_csv(file_obj):

    csv_data = pd.read_csv(CSV_PATH, names=colnames, delimiter=',')

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

    os.chdir(".\\")
    print("Creating train directories ...")
    for ISIC_class in class_names:
        if not os.path.exists(".\\data\\ISICset\\{}\\{}".format("train", ISIC_class)):
            pathlib.Path('.\\data\\ISICset\\{}\\{}'.format("train", ISIC_class)).mkdir(parents=True, exist_ok=True)
        else:
            print("Training directory already exist.")

    print("Creating test directories ...")
    for ISIC_class in class_names:
        if not os.path.exists(".\\data\\ISICset\\test\\{}".format(ISIC_class)):
            pathlib.Path('.\\data\\ISICset\\test\\{}'.format(ISIC_class)).mkdir(parents=True, exist_ok=True)
        else:
            print("Test directory already exist.")


'''
Copy actual images from ISIC dataset according to categories to
their own training directories
'''
def find_and_copy_images_to_training(current_list, folder_ext):
    print("Copying Image files to {} folder ... ".format(folder_ext))

    dst_dir = "data/ISICset/train/{}/".format(folder_ext)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in current_list:
        for file in glob.glob(IMAGES_FOLDER + "\\" + item + '*', recursive=True):
            if file not in glob.glob(dst_dir + "\\" + item + '*', recursive=True):
                shutil.copy(file, dst_dir)
                dst_file = os.path.join(dst_dir, file)
                new_dst_file_name = os.path.join(dst_dir, file)
                os.rename(dst_file, new_dst_file_name)

    print("Files copied >> {}".format(folder_ext))
    return

def move_images_to_test(training_dir, test_dir):

    files_to_move = []

    '''
        Moving 25 images from each training folder to respective
        test folder - Total of 175 test images.
    '''
    for dir in next(os.walk(training_dir))[1]:
        print("Moving from dir: ", dir)
        total_files_dir = get_all_images_in_dir(os.path.join(training_dir, dir))
        print(len(total_files_dir))
        for i in range(25):
            rnd_item = total_files_dir[random.randint(0, len(total_files_dir) - 1)]
            print("RND item: ", rnd_item)
            files_to_move.append(rnd_item)
            total_files_dir.remove(rnd_item)

        for image in files_to_move:
            if os.path.isfile(os.path.join(training_dir, dir, image)):
                print("image - ",image)
                shutil.move(os.path.join(training_dir, dir, image), os.path.join(test_dir, dir, image))




# Examples for rename and replace fun:
# rename(r"./data/ISICset/train/df/", r'*_*.jpg', r'df%s')
# replace(r"./data/ISICset/train/df/")
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
        rename('.\\data\\ISICset\\train\\' + "\\" + dir, r'*.jpg', r'{}%s'.format(dir))
        replace('.\\data\\ISICset\\train\\' + "\\" + dir)

def total_images_copied(training_dir):
    total = 0
    for root, dirs, files in os.walk(training_dir):
        total += len(files)

    print("Total images copied across categories :", total)


'''
    Fun that resizes training image size from base 600x450
    to 200x150 pixels.
    
'''
#TODO - choose image size dinamically
def resize_training_examples(training_dir):
    basewidth = 600
    for dir in next(os.walk(training_dir))[1]:
        print("RESIZING... : ", os.path.join(training_dir, dir))
        for dirname, dirs, files in os.walk(os.path.join(training_dir, dir)):
            for item in files:
                if os.path.isfile(os.path.join(training_dir, dir, item)):
                    im = Image.open(os.path.join(training_dir, dir, item))
                    x, y = im.size
                    new_dims = (200, 150)
                    output = im.resize(new_dims, Image.ANTIALIAS)
                    output_filename = os.path.join(training_dir, dir, item)
                    output.save(output_filename, "JPEG", quality=95)


'''
    Print out training and test set size.
'''
def print_dataset_stats():

    training_dir = os.environ.get("APP_TRAIN_DIR")
    test_dir = os.environ.get("APP_TEST_DIR")
    dataset_dict_train={}
    dataset_dict_test={}
    total_train = 0
    total_test = 0

    for dir in next(os.walk(training_dir))[1]:
        total_files = len([name for name in os.listdir(os.path.join(training_dir, dir)) if os.path.isfile(os.path.join(training_dir, dir, name))])
        total_train += total_files
        dataset_dict_train[dir] = total_files

    for dir in next(os.walk(test_dir))[1]:
        total_files = len([name for name in os.listdir(os.path.join(test_dir, dir)) if os.path.isfile(os.path.join(test_dir, dir, name))])
        total_test += total_files
        dataset_dict_test[dir] = total_files


    print("\nTraining dir stats:")
    print("------------------------------------")
    print(dataset_dict_train)
    print("Total num. of files in TRAIN: ", total_train)

    print("\nTest dir stats:")
    print("------------------------------------")
    print(dataset_dict_test)
    print("Total num. of files in TEST: ", total_test)



if __name__ == "__main__":
    CSV_PATH = os.environ.get("GT_TRAINING_DIR")
    with open(CSV_PATH, "rt") as f_obj:
        pandas_reader_csv(f_obj)
        create_directories()
        count_total_images(IMAGES_FOLDER)
        '''
            Copy all images to data directory
            - uncomment each line for categorizing, copying and resizing
        '''
        #find_and_copy_images_to_training(list_VASC, "vasc")
        #find_and_copy_images_to_training(list_NV, "nv")
        #find_and_copy_images_to_training(list_MEL, "mel")
        #find_and_copy_images_to_training(list_DF, "df")
        #find_and_copy_images_to_training(list_BKL, "bkl")
        #find_and_copy_images_to_training(list_BCC, "bcc")
        #find_and_copy_images_to_training(list_AKIEC, "akiec")


        #replace_and_rename('.\\data\\ISICset\\train\\')

        #resize_training_examples(".\\data\\ISICset\\train\\")

        #print_dataset_stats()

        #move_images_to_test(".\\data\\ISICset\\train\\", ".\\data\\ISICset\\test\\")

        print_dataset_stats()
