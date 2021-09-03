### This file contains a bunch of global variables, which are used by few other scripts
import os

# specifying project path (to your repo add /local_work/all_imgs)
# we stored metadata in '/local_work' and all the images in '/local_work/all_imgs'
# however, paths can also be specified inside training scripts
project_path = os.path.join(os.path.abspath(os.getcwd()), 'local_work')
imgs_path = os.path.join(project_path, 'all_imgs')
file_imgs = ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip']
file_imgs_metadata = 'HAM10000_metadata.csv'

# if one would like to compare model with use of different imbalance ratios, they all need to be specified here
# knowing the list before the training is required to ensure drawing similar-sized datasets (for better comparison)
imb_ratios = [1, 10, 100]

# dictionary of data labels
classes = {
    4: ('nv', ' melanocytic nevi'),
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc', ' basal cell carcinoma'),
    5: ('vasc', ' pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}

# breakpoints for binning of age - such mapped age can later be more effectively used for stratification
age_mapping = {
    50.0: '<0;50>', 999.0: '<50;inf)'
}
