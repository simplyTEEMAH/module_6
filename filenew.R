install.packages("keras")
library(keras)
install_keras()
fashion_mnist<-dataset_fashion_mnist()
x_train<-fashion_mnist$train$x
y_train<-fashion_mnist$train$y
x_test<-fashion_mnist$test$x
y_test<-fashion_mnist$test$y
# normalise the dataset
x_train <- x_train / 255
x_test <- x_test / 255
# Reshape data to correctly process as 2D images with a single channel
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
# One-hot encode the labels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
# create a sequential model
model<-keras_model_sequential()
# add layers to the model
model %>%
layer_conv_2d (filters=32, kernel_size=c(3,3), activation="relu", input_shape= c(28,28,1)) %>%
layer_conv_2d (filters=32, kernel_size=c(3,3), activation="relu") %>%
layer_conv_2d (filters=64, kernel_size=c(3,3), activation="relu") %>%
layer_max_pooling_2d (pool_size= c(2,2)) %>%
layer_conv_2d (filters=64, kernel_size=c(3,3), activation="relu") %>%
layer_conv_2d (filters=64, kernel_size=c(3,3), activation="relu") %>%
layer_max_pooling_2d (pool_size= c(2,2)) %>%
layer_conv_2d (filters=128, kernel_size=c(3,3), activation="relu") %>%
layer_dropout (rate=0.5) %>%
layer_flatten() %>%
layer_dense (units=10, activation="softmax")
# compile the model
model %>% compile (
    loss= "categorical_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
)
# train the model
history <- model %>% fit (
    x_train, y_train,
    batch_size = 64,
    epochs = 10,
    validation_data = list (x_test, y_test)
)
# the model has a 92.69% accuracy and a 91.34% validation accuracy    
set.seed(123)
# select two images randomly from the test images
sample_indices <- sample(1:nrow(x_test), 2)
# select the images
images <- x_test[sample_indices, , , 1]
# convert the encoded labels back to class labels
class_labels <- apply(y_test[sample_indices, ], 1, which.max) - 1
# make predictions for the randomly selected images
predictions <- model %>% predict(images)
# match selected images with its class
predicted_classes <- apply(predictions, 1, which.max) - 1
# display the images
for (i in 1:2) {
  img <- images[i, , ]
  class_label <- class_labels[i]
  predicted_label <- predicted_classes[i]
  # display the predicted images
  image(1:28, 1:28, t(img), col = gray.colors(255), axes = FALSE)
  # Add title showing the class and predicted labels
  title(paste("Class: ", class_label, " Pred: ", predicted_label), cex.main = 0.8)
}

