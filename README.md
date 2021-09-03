# ldam_str_bn

### Setup

---

The setup below works on a UNIX like system. Windows should work in a similar fashion. Just give it a quick google.
```bash
python3 -m venv <directory name>
source <directory name>/bin/activate
pip install -r requirements.txt
```
The dataset should be stored in a folder called `local_work` and all images should reside is a child folder called `all_imgs`. These names can also be adjusted in the config file. You can read more about the dataset in the corresponding section below.



### Datasets (HAM10000)

---

https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

With 7 columns :  `lesion_id`, `image_id`, `dx`, `dx__type`, `age`, `sex`, `localization `

ex) [HAM_0000118, ISIC_0027419 ,bkl, histo, 80.0, male, scalp]



### Topic & Tasks

---

When it comes to dealing with heavily imbalanced dataset, we focused on two approaches: __Label-distribution-aware loss function(LDAM)__ and __stratified batch normalization__.

 * Label-distribution-aware loss function(LDAM)
    * It encourages minority classes to have larger margins.
    * Introduced by this paper: https://arxiv.org/pdf/1906.07413.pdf
 * Stratified Batch Normalization
    * First layer of the net is being normalized separately for different stratification classes. For example, if sex and age_mapped are dimensions used for stratification, there will be 6 stratification classes (cartesian of (male,female,unknown) and (<=50, >50)).
    * Each stratification class uses its own set of gammas and betas
    * The underlying idea of stratification is the assumption that for different stratification classes, distributions of labels differ significantly. Therefore, they should be made even before being fed to the network.

We artificially made medical imaging dataset to be highly imbalanced (with different imbalance ratios). `strat_data_generator` `utils_sc.draw_data()` implement this functionality. Then, we implemented stratified batch normalization (`models.strat_bn_simplified`) within a ResNet model (`models.resnet`) with use of Label-Distribution-Aware loss function (`losses`). In the end, we perform unit tests with `unittest` python module for both the loss function and stratified batch normalization to check if they function correctly.



### Challenges

---

1. Finding a suitable network architecture
2. Deciding on what dimensions do we stratify - choice of features and dealing with binary data transformation.
3. Implementation of stratified batch normalization
   * Understanding the concept and original Tensorflow BN implementation
   * Dealing with parameters in new shapes for both training and non-training modes (i.e. updating/using `moving_mean`, `moving_variance`,  `beta`, `gamma`) 
4. Converting LDAM loss function from PyTorch to Tensorflow
   * Understanding the concept of LDAM in general
   * Dealing with different data structures & methods 



### Team's contribution

---

1. Data Preprocessing - implemented our own data generator `strat_data_generator` and `utils_sc`

2. Implemented LDAM loss in Tensorflow (`losses`)

3. Implemented of stratified batch normalization with ResNet model (`models.strat_bn_simplified`, `models.resnet`)

4. Unit tests for LDAM loss and Stratified Batch Normalization with `unittest`:
   * LDAM loss - compare both pytorch LDAM loss and tensorflow LDAM loss unit by unit
   * Stratified Batch Normalization



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

  Idea of batch normalization in general :

  * [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

  Loosely connected paper (explains the idea of stratified batch normalization) :

  * [(PDF) Cross-Subject EEG-Based Emotion Recognition through Neural Networks with Stratified Normalization](https://www.researchgate.net/publication/344377115_Cross-Subject_EEG-Based_Emotion_Recognition_through_Neural_Networks_with_Stratified_Normalization)

* LDAM loss

  Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss :

  * https://arxiv.org/pdf/1906.07413.pdf

  * https://github.com/kaidic/LDAM-DRW/blob/master/losses.py (Pytorch implementation of the authors) 

* Data Generator

  Inspired by this implementation :

  * https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

---

#### Team ‘Weißwürstchen’

__Seunghee Jeong [seunghee6022@gmail.com](mailto:seunghee6022@gmail.com)__

__Nick Stracke [nick.stracke@web.de](mailto:nick.stracke@web.de)__

__Karol Urbańczyk [karurb92@gmail.com](mailto:karurb92@gmail.com)__

---

