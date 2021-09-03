### a bunch of util functions that are later used by training scripts

import zipfile
import pandas as pd
import numpy as np
import math
import config_sc

# unzip your images
def unzip_imgs(data_path):
  for file_img in config_sc.file_imgs:
    with zipfile.ZipFile(f'{data_path}/{file_img}', 'r') as zip_ref:
      zip_ref.extractall(f'{data_path}')

# based on imbalance_ratios (config_sc), this function checks what is the maximum common n of images that can be drawn to ensure similar size of datasets for different models
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

# this function performs all the necessary steps that are needed to prepare the data for later usage by data generator
def draw_data(metadf, imb_ratio, strat_dims=['sex', 'age_mapped'], train_split = 0.8):
  # for each stratification dimension filter out all empty rows (e.g. some dont have age)
  df = metadf
  for dim in strat_dims:
    df = df[df[dim].notnull()]

  # define to which stratification class image belongs. e.g. for ['sex', 'age_mapped'] there are six classes (cartesian of ['male', 'female', 'unknown'] and ['<0;50>', '<50;inf)']) 
  if len(strat_dims)==0:
    df['strat_class'] = 0
    strat_classes_num = 1
  else:
    df_strat = df[strat_dims].groupby(strat_dims).size().reset_index().rename(columns={0:'strat_class'})
    df_strat['strat_class'] = df_strat.index
    df = df.merge(df_strat, on=strat_dims)
    strat_classes_num = len(df_strat)

  # given whole uploaded dataset, check how big part of it can be used (to ensure common size set for different imbalance ratios) 
  y='dx'
  max_n = check_max_n(df, y)

  # define what is the biggest y class
  main_class = df[['image_id', y]].groupby(y).agg('count').sort_values(by='image_id', ascending=False).index[0]

  # assign numbers to y labels according to dictionary
  dict_reverse = dict([value[0], key] for key, value in config_sc.classes.items())
  df = df.merge(pd.DataFrame(dict_reverse.items()), left_on='dx', right_on=0)
  
  # define list of counts of every y class. this is later used by LDAM loss
  cls_num_list = list(df.groupby(1).size())

  # draws data according to given imbalance ratio (df_main is the biggest class, df_rest is all others)
  # general: don't use imbalance ratio which is outside of range of what is defined in config_sc.imb_ratios
  df_main = df[df[y]==main_class].sample(math.floor(1.0*imb_ratio/(imb_ratio+1)*max_n))
  df_rest = df[df[y]!=main_class].sample(math.floor(1.0/(imb_ratio+1)*max_n))
  df_drawn = pd.concat([df_main, df_rest])

  # defines dictionary of image ids and y labels (later used by data generator)
  labels = pd.Series(df_drawn[1].values, index = df_drawn['image_id']).to_dict()

  # takes only the columns subset that is needed later
  columns = ['image_id', 'strat_class']
  df_drawn = df_drawn[columns]

  # splits data into training and validation
  msk = np.random.rand(len(df_drawn)) < train_split
  df_train = df_drawn[msk]
  df_val = df_drawn[~msk]

  # such a form (list) is required later by data generator
  data_train = df_train.values.tolist()
  data_val = df_val.values.tolist()

  return data_train, data_val, labels, strat_classes_num, cls_num_list

# loads metadata csv into pandas. adds age bucketing.
# general: usage of this assumes that metadata only has training+validation imgs. test_data for potential model comparison should be excluded first
def load_metadf(data_path):
  metadf = pd.read_csv(f'{data_path}/{config_sc.file_imgs_metadata}')
  break_points = [-1.0] + sorted(config_sc.age_mapping)
  age_labels = [config_sc.age_mapping[value] for value in sorted(config_sc.age_mapping)]
  metadf['age_mapped'] = pd.cut(metadf['age'], bins=break_points, labels=age_labels)
  return metadf