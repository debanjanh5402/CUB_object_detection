## CUB Object Detection Project: A Deep Learning Approach

This project is a detailed implementation of an object detection model designed to identify and localize 200 different bird species from the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset. The entire workflow, from data preprocessing and augmentation to model training and evaluation, is meticulously documented and implemented using the **TensorFlow** framework.

**Note:** The trained model weights are not included in this repository due to their large size. The provided notebook contains all the code required to train the model from scratch.

-----

### Project Structure

  * `fair.ipynb`: The primary Jupyter notebook that contains all the source code for the project, including data pipeline, model architecture, training, and evaluation.
  * `training_history_EfficientNet.json`: A JSON file that logs the performance metrics (loss, accuracy, and IoU) for both the training and validation sets across all epochs.
  * `plots.png`: A visual representation of the training and validation metrics.
  * `pred1.png`, `pred2.png`: Example images showcasing the model's predictions on test data.

-----

### Methodology

#### Data Preprocessing and Augmentation

A robust data pipeline is essential for training deep learning models. This project's pipeline uses `tf.data.Dataset` for efficient data handling and includes a comprehensive set of data augmentation techniques to prevent overfitting.

1.  **Data Loading**: A Pandas DataFrame is created from the dataset's text files to manage image paths, class labels, and bounding box coordinates.
2.  **Global Parameters**: The model operates on a fixed image size of **224x224 pixels** with a batch size of **32**, and was trained for **100 epochs**.
3.  **Data Augmentation**: To improve model generalization, the following augmentations are applied to the training data in the `augment_data` function:
      * **Random Flip**: Horizontally flips images and updates bounding box coordinates accordingly.
      * **Random Rotation**: Rotates images by a random angle (up to 5 degrees) and recalculates the new bounding box coordinates.
      * **Random Scaling**: Randomly scales the image (between 95% and 105% of its original size) and adjusts the bounding boxes.
      * **Random Erasing**: Randomly zeroes out a small portion of the image.
      * **Color Augmentation**: Randomly adjusts brightness, contrast, hue, and saturation.
4.  **Normalization**: Images are resized, normalized to a `[0, 255]` range, and then preprocessed using `efficientnet_v2.preprocess_input` to match the requirements of the EfficientNetV2 model.

#### Model Architecture

The model is a custom object detector built on top of a pre-trained feature extractor.

  * **Feature Extractor**: The model uses a non-trainable **EfficientNetV2S** base, pre-trained on the ImageNet dataset, to extract powerful features from the images.
  * **Detector Head**: Two dense layers are added on top of the feature extractor to predict the class and bounding box coordinates.
      * **Classification Head**: A `Dense` layer with `softmax` activation predicts the probability of each of the 200 bird classes.
      * **Bounding Box Regression Head**: A `Dense` layer with `sigmoid` activation predicts the four normalized coordinates of the bounding box.

#### Loss Function and Metrics

The model is compiled with a custom loss function and specific metrics to evaluate performance effectively.

  * **Loss Function**:
      * **Classification Loss**: `SparseCategoricalCrossentropy` is used for the bird classification task.
      * **Bounding Box Loss**: A custom `focal_eiou` loss function, which combines **Focal Loss** with **EIoU (Extended Intersection over Union)**, is used to accurately predict the bounding boxes.
  * **Metrics**:
      * **Classification Metrics**: `SparseCategoricalAccuracy` and `SparseTopKCategoricalAccuracy` (top-100) track the classification performance.
      * **Bounding Box Metric**: A custom `iou` metric calculates the Intersection over Union to measure the accuracy of the predicted bounding boxes.
  * **Optimizer**: The model is trained using the **AdamW** optimizer with a **Cosine Decay Restarts** learning rate schedule.

-----

### Results

The model was trained for 100 epochs with early stopping after 10 epochs without improvement in validation loss, restoring the best weights.

The final evaluation on the test dataset yielded the following results:

  * **Mean IoU**: `0.5741`
  * **Accuracy**: `0.7050`
  * **Top-100 Accuracy**: `0.9991`

#### Training History

The `training_history_EfficientNet.json` file contains the full log of training and validation metrics. The provided `plots.png` file visualizes these metrics over the training epochs.

#### Example Predictions

The `prediction.png` image provides a visual representation of the model's performance on a few test images. The plots show the ground truth (GT) bounding box in **green** and the predicted (PR) bounding box in **red**, along with the predicted class, confidence, and IoU score for each prediction.

#### Classification Report

The model's classification performance for each of the 200 classes is detailed in the classification report below.

```
                                precision    recall  f1-score   support

        Black_footed_Albatross       0.80      0.80      0.80        30
              Laysan_Albatross       0.70      0.77      0.73        30
               Sooty_Albatross       0.63      0.86      0.73        28
             Groove_billed_Ani       0.85      0.93      0.89        30
                Crested_Auklet       0.85      0.79      0.81        14
                  Least_Auklet       0.77      0.91      0.83        11
               Parakeet_Auklet       0.96      0.96      0.96        23
             Rhinoceros_Auklet       0.74      0.78      0.76        18
              Brewer_Blackbird       0.32      0.34      0.33        29
          Red_winged_Blackbird       0.96      0.80      0.87        30
               Rusty_Blackbird       0.74      0.57      0.64        30
       Yellow_headed_Blackbird       0.96      0.92      0.94        26
                      Bobolink       0.73      0.80      0.76        30
                Indigo_Bunting       0.81      0.87      0.84        30
                Lazuli_Bunting       0.79      0.79      0.79        28
               Painted_Bunting       0.96      0.86      0.91        28
                      Cardinal       0.90      0.96      0.93        27
               Spotted_Catbird       1.00      1.00      1.00        15
                  Gray_Catbird       0.92      0.83      0.87        29
          Yellow_breasted_Chat       0.65      0.69      0.67        29
                Eastern_Towhee       0.88      0.77      0.82        30
              Chuck_will_Widow       0.65      0.58      0.61        26
              Brandt_Cormorant       0.59      0.55      0.57        29
           Red_faced_Cormorant       0.75      0.82      0.78        22
             Pelagic_Cormorant       0.43      0.40      0.41        30
               Bronzed_Cowbird       0.67      0.60      0.63        30
                 Shiny_Cowbird       0.38      0.53      0.44        30
                 Brown_Creeper       1.00      0.90      0.95        29
                 American_Crow       0.53      0.33      0.41        30
                     Fish_Crow       0.35      0.47      0.40        30
           Black_billed_Cuckoo       0.63      0.73      0.68        30
               Mangrove_Cuckoo       0.64      0.61      0.62        23
          Yellow_billed_Cuckoo       0.69      0.62      0.65        29
       Gray_crowned_Rosy_Finch       0.96      0.79      0.87        29
                  Purple_Finch       0.88      0.93      0.90        30
              Northern_Flicker       0.93      0.83      0.88        30
            Acadian_Flycatcher       0.52      0.48      0.50        29
      Great_Crested_Flycatcher       0.56      0.63      0.59        30
              Least_Flycatcher       0.32      0.38      0.35        29
        Olive_sided_Flycatcher       0.42      0.43      0.43        30
     Scissor_tailed_Flycatcher       0.76      0.73      0.75        30
          Vermilion_Flycatcher       0.97      0.93      0.95        30
     Yellow_bellied_Flycatcher       0.54      0.45      0.49        29
                   Frigatebird       0.93      0.87      0.90        30
               Northern_Fulmar       0.66      0.70      0.68        30
                       Gadwall       0.90      0.90      0.90        30
            American_Goldfinch       0.74      0.97      0.84        30
            European_Goldfinch       0.88      0.93      0.90        30
           Boat_tailed_Grackle       0.58      0.47      0.52        30
                   Eared_Grebe       0.70      0.77      0.73        30
                  Horned_Grebe       0.69      0.67      0.68        30
             Pied_billed_Grebe       0.88      0.93      0.90        30
                 Western_Grebe       1.00      0.87      0.93        30
                 Blue_Grosbeak       0.76      0.83      0.79        30
              Evening_Grosbeak       0.94      0.97      0.95        30
                 Pine_Grosbeak       0.86      0.80      0.83        30
        Rose_breasted_Grosbeak       0.92      0.80      0.86        30
              Pigeon_Guillemot       0.88      0.82      0.85        28
               California_Gull       0.47      0.23      0.31        30
          Glaucous_winged_Gull       0.53      0.34      0.42        29
                 Heermann_Gull       0.91      0.97      0.94        30
                  Herring_Gull       0.34      0.37      0.35        30
                    Ivory_Gull       0.78      0.83      0.81        30
              Ring_billed_Gull       0.47      0.57      0.52        30
             Slaty_backed_Gull       0.45      0.45      0.45        20
                  Western_Gull       0.36      0.53      0.43        30
              Anna_Hummingbird       0.63      0.63      0.63        30
     Ruby_throated_Hummingbird       0.72      0.70      0.71        30
            Rufous_Hummingbird       0.81      0.83      0.82        30
               Green_Violetear       0.94      1.00      0.97        30
            Long_tailed_Jaeger       0.46      0.63      0.54        30
               Pomarine_Jaeger       0.74      0.47      0.57        30
                      Blue_Jay       0.94      0.97      0.95        30
                   Florida_Jay       0.86      1.00      0.92        30
                     Green_Jay       1.00      0.89      0.94        27
               Dark_eyed_Junco       0.78      0.97      0.87        30
             Tropical_Kingbird       0.71      0.57      0.63        30
                 Gray_Kingbird       0.75      0.72      0.74        29
             Belted_Kingfisher       0.66      0.63      0.64        30
              Green_Kingfisher       0.78      0.70      0.74        30
               Pied_Kingfisher       0.76      0.93      0.84        30
             Ringed_Kingfisher       0.67      0.73      0.70        30
     White_breasted_Kingfisher       1.00      0.90      0.95        30
          Red_legged_Kittiwake       0.77      0.74      0.76        23
                   Horned_Lark       0.93      0.83      0.88        30
                  Pacific_Loon       0.92      0.77      0.84        30
                       Mallard       0.97      0.97      0.97        30
            Western_Meadowlark       0.96      0.83      0.89        30
              Hooded_Merganser       1.00      0.77      0.87        30
        Red_breasted_Merganser       0.74      0.93      0.82        30
                   Mockingbird       0.48      0.53      0.51        30
                     Nighthawk       0.88      0.77      0.82        30
              Clark_Nutcracker       0.94      0.97      0.95        30
       White_breasted_Nuthatch       0.93      0.93      0.93        30
              Baltimore_Oriole       0.85      0.73      0.79        30
                 Hooded_Oriole       1.00      0.63      0.78        30
                Orchard_Oriole       0.67      0.69      0.68        29
                  Scott_Oriole       0.82      0.90      0.86        30
                      Ovenbird       0.71      0.73      0.72        30
                 Brown_Pelican       0.97      1.00      0.98        30
                 White_Pelican       0.95      0.95      0.95        20
            Western_Wood_Pewee       0.30      0.27      0.28        30
                      Sayornis       0.45      0.50      0.48        30
                American_Pipit       0.56      0.63      0.59        30
                Whip_poor_Will       0.50      0.79      0.61        19
                 Horned_Puffin       0.96      0.87      0.91        30
                  Common_Raven       0.48      0.40      0.44        30
            White_necked_Raven       0.79      0.73      0.76        30
             American_Redstart       0.77      0.80      0.79        30
                     Geococcyx       0.78      0.97      0.87        30
             Loggerhead_Shrike       0.66      0.63      0.64        30
             Great_Grey_Shrike       0.68      0.63      0.66        30
                 Baird_Sparrow       0.65      0.65      0.65        20
        Black_throated_Sparrow       0.73      0.90      0.81        30
                Brewer_Sparrow       0.50      0.52      0.51        29
              Chipping_Sparrow       0.53      0.53      0.53        30
          Clay_colored_Sparrow       0.32      0.41      0.36        29
                 House_Sparrow       0.50      0.50      0.50        30
                 Field_Sparrow       0.56      0.52      0.54        29
                   Fox_Sparrow       0.84      0.53      0.65        30
           Grasshopper_Sparrow       0.55      0.70      0.62        30
                Harris_Sparrow       0.81      0.73      0.77        30
               Henslow_Sparrow       0.50      0.70      0.58        30
              Le_Conte_Sparrow       0.69      0.62      0.65        29
               Lincoln_Sparrow       0.50      0.55      0.52        29
   Nelson_Sharp_tailed_Sparrow       0.56      0.47      0.51        30
              Savannah_Sparrow       0.41      0.50      0.45        30
               Seaside_Sparrow       0.63      0.57      0.60        30
                  Song_Sparrow       0.62      0.43      0.51        30
                  Tree_Sparrow       0.50      0.43      0.46        30
                Vesper_Sparrow       0.56      0.63      0.59        30
         White_crowned_Sparrow       0.83      0.80      0.81        30
        White_throated_Sparrow       0.68      0.83      0.75        30
          Cape_Glossy_Starling       0.77      0.90      0.83        30
                  Bank_Swallow       0.66      0.63      0.64        30
                  Barn_Swallow       0.55      0.60      0.57        30
                 Cliff_Swallow       0.61      0.57      0.59        30
                  Tree_Swallow       0.95      0.67      0.78        30
               Scarlet_Tanager       0.92      0.73      0.81        30
                Summer_Tanager       0.85      0.97      0.91        30
                    Artic_Tern       0.60      0.62      0.61        29
                    Black_Tern       0.65      0.50      0.57        30
                  Caspian_Tern       0.52      0.40      0.45        30
                   Common_Tern       0.21      0.27      0.24        30
                  Elegant_Tern       0.47      0.57      0.52        30
                 Forsters_Tern       0.41      0.23      0.30        30
                    Least_Tern       0.73      0.73      0.73        30
           Green_tailed_Towhee       0.86      0.83      0.85        30
                Brown_Thrasher       0.92      0.76      0.83        29
                 Sage_Thrasher       0.73      0.73      0.73        30
            Black_capped_Vireo       0.67      0.76      0.71        21
             Blue_headed_Vireo       0.76      0.53      0.63        30
            Philadelphia_Vireo       0.68      0.59      0.63        29
                Red_eyed_Vireo       0.50      0.60      0.55        30
                Warbling_Vireo       0.41      0.53      0.46        30
              White_eyed_Vireo       0.35      0.37      0.36        30
         Yellow_throated_Vireo       0.54      0.45      0.49        29
          Bay_breasted_Warbler       0.88      0.93      0.90        30
       Black_and_white_Warbler       0.93      0.93      0.93        30
   Black_throated_Blue_Warbler       0.79      0.90      0.84        29
           Blue_winged_Warbler       0.54      0.73      0.62        30
                Canada_Warbler       0.75      0.70      0.72        30
              Cape_May_Warbler       0.69      0.80      0.74        30
              Cerulean_Warbler       0.86      0.80      0.83        30
        Chestnut_sided_Warbler       0.71      0.67      0.69        30
         Golden_winged_Warbler       0.92      0.79      0.85        29
                Hooded_Warbler       0.85      0.57      0.68        30
              Kentucky_Warbler       0.83      0.86      0.85        29
              Magnolia_Warbler       0.76      0.86      0.81        29
              Mourning_Warbler       0.90      0.60      0.72        30
                Myrtle_Warbler       0.68      0.50      0.58        30
             Nashville_Warbler       0.36      0.67      0.47        30
        Orange_crowned_Warbler       0.50      0.47      0.48        30
                  Palm_Warbler       0.50      0.67      0.57        30
                  Pine_Warbler       0.62      0.53      0.57        30
               Prairie_Warbler       0.83      0.67      0.74        30
          Prothonotary_Warbler       0.70      0.77      0.73        30
              Swainson_Warbler       0.71      0.65      0.68        26
             Tennessee_Warbler       0.37      0.34      0.36        29
                Wilson_Warbler       0.59      0.80      0.68        30
           Worm_eating_Warbler       0.79      0.79      0.79        29
                Yellow_Warbler       0.88      0.70      0.78        30
          Northern_Waterthrush       0.76      0.63      0.69        30
         Louisiana_Waterthrush       0.81      0.83      0.82        30
              Bohemian_Waxwing       0.86      0.80      0.83        30
                 Cedar_Waxwing       0.81      0.83      0.82        30
American_Three_toed_Woodpecker       1.00      0.80      0.89        20
           Pileated_Woodpecker       0.93      0.87      0.90        30
        Red_bellied_Woodpecker       0.83      0.97      0.89        30
       Red_cockaded_Woodpecker       0.97      0.97      0.97        29
         Red_headed_Woodpecker       0.86      0.83      0.85        30
              Downy_Woodpecker       0.86      0.83      0.85        30
                   Bewick_Wren       0.67      0.67      0.67        30
                   Cactus_Wren       0.83      0.63      0.72        30
                 Carolina_Wren       0.66      0.63      0.64        30
                    House_Wren       0.63      0.57      0.60        30
                    Marsh_Wren       0.68      0.70      0.69        30
                     Rock_Wren       0.93      0.87      0.90        30
                   Winter_Wren       0.77      0.90      0.83        30
           Common_Yellowthroat       0.79      0.90      0.84        30

                      accuracy                           0.71      5794
                     macro avg       0.72      0.71      0.71      5794
                  weighted avg       0.72      0.71      0.71      5794
```
