#utils
from google.colab import drive
import zipfile
import pandas as pd
import config_sc

def connect_gdrive():
  drive.mount('/content/drive/')

def unzip_imgs():
  for file_img in file_imgs:
    with zipfile.ZipFile(f'{project_path}/{file_img}', 'r') as zip_ref:
      zip_ref.extractall(f'{project_path}/{imgs}')

def draw_data(metadf, imb_ratio, strat_dims, y='dx'):
  #TODO: for each strat_dim filter out all null/empty/whatever (need to check types) rows (e.g. some dont have age). save to df
  df = metadf
  total_count = df.shape[0]
  main_class_count = df[['image_id', y]].groupby(y).agg('count').sort_values(by='image_id', ascending=False)['image_id'][0]
  if main_class_count >= imb_ratio * (total_count - main_class_count):
    #TODO: take all rows from other classes and sample imb_ratio * (total_count - main_class_count) rows from main class
    pass
  else:
    #TODO: take all rows from main class and sample floor(main_class_count / imb_ratio) rows from other classes
    pass
  return df

def load_metadf():
  metadf = pd.read_csv(f'{config_sc.project_path}/{config_sc.file_imgs_metadata}')
  metadf['dx_alternative'] = metadf['dx'].map(config_sc.y_mapping)
  break_points = [-1.0] + sorted(config_sc.age_mapping)
  labels = [config_sc.age_mapping[value] for value in sorted(config_sc.age_mapping)]
  metadf['age_mapped'] = pd.cut(metadf['age'], bins=break_points, labels=labels)
  return metadf