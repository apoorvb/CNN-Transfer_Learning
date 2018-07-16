import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')

type(model)
model.summary()
model.layers.pop()
model.summary()
vgg = Sequential()
vgg.add(model)
vgg.summary()

for layer in vgg.layers:
    layer.trainable = False

#incpn.add(Flatten())
vgg.add(Dense(4096, activation = 'relu'))
vgg.add(Dense(2, input_shape=(1, ), activation = 'softmax'))

vgg.summary()

vgg.compile(Adam(lr = .0001), loss = 'categorical_crossentropy', metrics = ['accuracy'] )

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/apocobis/Downloads/Dataset/cat&dog/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 classes = ['dogs', 'cats'])

test_set = test_datagen.flow_from_directory('/home/apocobis/Downloads/Dataset/cat&dog/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            classes = ['dogs', 'cats'])

vgg.fit_generator(training_set, steps_per_epoch = 8000, epochs = 10, validation_data = test_set, validation_steps = 2000)


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/apocobis/Downloads/Dataset/cat&dog/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/home/apocobis/Downloads/Dataset/cat&dog/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 10, validation_data = test_set, validation_steps = 2000)












