#utils
#from google.colab import drive
import zipfile
import pandas as pd
import math
import config_sc

def connect_gdrive():
  drive.mount('/content/drive/')

def unzip_imgs():
  for file_img in config_sc.file_imgs:
    with zipfile.ZipFile(f'{config_sc.project_path}/{file_img}', 'r') as zip_ref:
      zip_ref.extractall(f'{config_sc.project_path}/{config_sc.imgs}')

#this checks what is the max n we can use for training, given imbalance ratios and the fact models should be trained on similarly sized sets
def check_max_n(df, y):
  total_count = df.shape[0]
  main_class_count = df[['image_id', y]].groupby(y).agg('count').sort_values(by='image_id', ascending=False)['image_id'][0]
  max_n = total_count
  for imb in config_sc.imb_ratios:
    if main_class_count >= imb * (total_count - main_class_count):
      max_local = (imb + 1) * (total_count - main_class_count)
    else:
      max_local = math.floor(1.0 * main_class_count / imb) + main_class_count
    if max_local < max_n:
      max_n = max_local
  return 1.0 * max_n

#this draws data according to given imbalance ratio
#dont use imb ratios outside of range of what is defined in config_sc.imb_ratios
def draw_data(metadf, imb_ratio, strat_dims=['sex', 'age_mapped'], y='dx'):
  #for each strat_dim filter out all empty rows (e.g. some dont have age)
  df = metadf
  for dim in strat_dims:
    df = df[df[dim].notnull()]

  max_n = check_max_n(df, y)
  main_class = df[['image_id', y]].groupby(y).agg('count').sort_values(by='image_id', ascending=False).index[0]

  df_main = df[df[y]==main_class].sample(math.floor(1.0*imb_ratio/(imb_ratio+1)*max_n))
  df_rest = df[df[y]!=main_class].sample(math.floor(1.0/(imb_ratio+1)*max_n))
  df_drawn = pd.concat([df_main, df_rest])

  return df_drawn

def load_metadf():
  metadf = pd.read_csv(f'{config_sc.project_path}/{config_sc.file_imgs_metadata}')
  metadf['dx_alternative'] = metadf['dx'].map(config_sc.y_mapping)
  break_points = [-1.0] + sorted(config_sc.age_mapping)
  labels = [config_sc.age_mapping[value] for value in sorted(config_sc.age_mapping)]
  metadf['age_mapped'] = pd.cut(metadf['age'], bins=break_points, labels=labels)
  return metadf