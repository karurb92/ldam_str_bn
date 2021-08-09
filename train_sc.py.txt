'''params for training will be:
y - (string) which y to use ('dx' or 'dx_alternative')
strat_dims - (list) of dimensions to stratify on (dx_type / sex / age_mapped / localization). maybe localization makes more sense because it affects photos the most? distr of ys is heavily dependent on sex and age
imb_ratio - (float) imbalance ratio (main class count / all other count)
etc. (all the learning rates, depth, batch size etc)
'''

from config_sc import *
from utils_sc import *
import pandas as pd

def main():
  #parse args here
  
  metadf = load_metadf()
  data = draw_data(metadf, imb_ratio, strat_dims, y)

  #further training from here

if __name__ == '__main__':
    main()