# config
project_path = '/content/drive/My Drive/Colab_Notebooks/LMU_appliedDL/skincancer'
file_imgs = ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip']
file_imgs_metadata = 'HAM10000_metadata.csv'
imgs = 'all_imgs'

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
