# config
import os

project_path = os.path.join(os.path.abspath(os.getcwd()), 'local_work')
imgs_path = os.path.join(project_path, 'all_imgs')
file_imgs = ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip']
file_imgs_metadata = 'HAM10000_metadata.csv'

# for this ratios we will train different models. it's used to assess n of rows to be drawn from the data (to ensure every model has same n)
imb_ratios = [1, 10, 100]

'''
to be deleted?
# in case we want to work on simplified problem. dict after short research, might need confirmation
y_mapping = {
    'bkl': 'benign',
    'nv': 'benign',
    'df': 'benign',
    'mel': 'malignant & deadly',
    'vasc': 'benign',
    'bcc': 'malignant & safe',
    'akiec': 'benign'
}
'''

classes = {
    4: ('nv', ' melanocytic nevi'),
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc', ' basal cell carcinoma'),
    5: ('vasc', ' pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}

# we could use also breakpoints at 30/40,60
age_mapping = {
    50.0: '<0;50>', 999.0: '<50;inf)'
}
