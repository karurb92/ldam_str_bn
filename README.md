# ldam_str_bn

### Setup

---

The setup below works on a UNIX like system. Windows should work in a similar fashion. Just give it a quick google.
```bash
python3 -m venv <directory name>
source <directory name>/bin/activate
pip install -r requirements.txt
```
The dataset should be stored in a foldler called `local_work` and all images should reside is a child folder called `all_imgs`. These names can also be adjusted in the config file. You can read more about the dataset in the corresponding section below.



### Datasets (HAM10000)

---

https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

With 7 columns :  `lesion_id`, `image_id`, `dx`, `dx__type`, `age`, `sex`, `localization `

ex) [HAM_0000118, ISIC_0027419 ,bkl, histo, 80.0, male, scalp]



### Topic & Tasks

---

When it comes to dealing with heavily imbalanced dataset, we focused on two approaches, __Label-distribution-aware loss function(LDAM)__ and __stratified batch normalization__. Because of the advantages of them toward imbalanced data. 

 * Label-distribution-aware loss function(LDAM)
    * model to have optimal trade-off between per-class margins by encouraging the minority classes to have large margins
 * Stratified Batch Normalization
    * compared to using batch normalization, significantly increases the accuracy by feature normalization per sex and localization? and by reducing the correlations

We made medical imaging dataset to be highly imbalanced with implemented `strat_data_generator` function. We implemented stratified batch normalization within a ResNet model and train it with label-distribution-aware loss function(LDAM). After the implementation, we perform unit test by `unittest` python module to the loss function and stratified batch normalization to check both function correctly.



### Challenges

---

1. Finding a suitable network architecture
2. Deciding on what dimensions we stratify - choice of features to be stratified and dealing with binary data trasformation.
3. Implementation of stratified batch normalization
   1. Understanding the concepts 
   2. Finding balance between stratification and dataset size
   3. dealing with parameters. ( i.e. saving `moving_mean`, `moving_variance`,  `beta`, `gamma` etc and and to update them in the next training) 
4. Converting LDAM loss function to tensorflow
   1. different type of data and function and its inputs
   2. Understanding the structures 
5. Actually training the models 
   1. Handling input data size/processing none errors to fit the model
   2. Dealing with minor errors while training



### Contribution

---

1. Data Preprocessing - implemented `strat_data_generator` to make imaging dataset imbalanced
   * add explanation if you want

2. Implementing LDAM loss with tensorflow

3. Implementation of stratified batch normalization with ResNet model

4. Unit tests for LDAM loss and Stratified Batch Normalization by `unittest` module of python
   * LDAM loss - compare both pytorch LDAM loss and tensorflow LDAM loss unit by unit
   * Stratified Batch Normalization

5. Meaningful results by each implementations and actual training 



### Results

---

* Imbalanced Dataset

  > Imbalance ratio : 10

  guess show some data distribution image here.

* Stratified Batch Normalization

  > batch size 32

  Changes of `beta`, `gamma`, `moving_mean`, `movng_variance`

  <img src=".\readme_images\training_batch32.jpg">

* Training

  >  Decreasing the loss

  show kind of line graph of loss changes

  > Changing Epoch/batch size

  > Imbalanced data vs Balanced data

* Test

  > Correct prediction with the wrong labels

  show wrong label data sample and show the output of it.

  >  Test with real data?

### References

---

* Stratified Batch Normalization

  Loosely connected paper (explains the idea of stratified batch normalization) :

  [(PDF) Cross-Subject EEG-Based Emotion Recognition through Neural Networks with Stratified Normalization](https://www.researchgate.net/publication/344377115_Cross-Subject_EEG-Based_Emotion_Recognition_through_Neural_Networks_with_Stratified_Normalization)

* LDAM loss

  Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss :

  https://arxiv.org/pdf/1906.07413.pdf

  https://github.com/kaidic/LDAM-DRW/blob/master/losses.py (Pytorch ver.) 



---

#### Team ‘Weißwürstchen’

__Seunghee Jeong [seunghee6022@gmail.com](mailto:seunghee6022@gmail.com)__

__Nick Stracke [nick.stracke@web.de](mailto:nick.stracke@web.de)__

__Karol Urbańczyk [karurb92@gmail.com](mailto:karurb92@gmail.com)__



---

