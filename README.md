# CNN-Transfer_Learning
2 different versions of Image Classifier were used.
Metric used to compare their performance was validation accuracy.

# Transfer Learning : VGG16
The model can be imported from keras itself.
Last FC layers were popped
New trainable FC layers are added
All previous layers were frozen
Validation accuracy : >90%

# CNN Image classifier
An ordinary CNN model was used
Model has more than 1 Conv layers
Validation Accuracy : 80%

# Conclusion
Transfer Learning proved to be more accurate with 90-100% accuracy
CNN models don't always have such high accuracy
Alot of time is wasted in choosing the hyper parameters. So many trials and errors
Whereas in the case of Transfer Learning, you always get such high-accuracy.
This is why Transfer Learning is the most effective method of solving computer vision problems.
