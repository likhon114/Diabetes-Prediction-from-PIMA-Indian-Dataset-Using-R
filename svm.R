# Load necessary libraries
library(caret)
library(e1071)

# Load the train and test datasets
train_set <- readRDS("train_set.rds")
test_set <- readRDS("test_set.rds")

# Set up 10-fold cross-validation
control <- trainControl(method = "cv", number = 10)

# Define a grid of C and sigma values to search over
tune_grid <- expand.grid(
  C = c(0.1, 1, 10, 100, 1000),   # Cost parameter
  sigma = c(0.01, 0.1, 0.5, 1)    # Sigma (gamma) parameter
)

# Train the SVM model using the radial kernel and grid search
svm_model_tuned <- train(
  diabetes ~ ., 
  data = train_set, 
  method = "svmRadial",            # Radial basis function kernel
  trControl = control,            # Cross-validation control
  tuneGrid = tune_grid,           # Grid of C and sigma
  preProcess = c("center", "scale") )

# Print the best model and parameters
print(svm_model_tuned)

# Get the best tuning parameters (C and sigma)
best_params <- svm_model_tuned$bestTune
cat("Best C:", best_params$C, "\n")
cat("Best sigma:", best_params$sigma, "\n")

# Evaluate the model on the test set
svm_predictions <- predict(svm_model_tuned, newdata = test_set)
confusion_matrix <- confusionMatrix(svm_predictions, test_set$diabetes)

# Print the accuracy on the test set
cat("Test set accuracy:", confusion_matrix$overall["Accuracy"], "\n")


library(caret)
library(ggplot2)

# Make predictions on the test set with the best model
predictions <- predict(svm_model_tuned, newdata = test_set)

# Calculate confusion matrix
confusion_matrix <- confusionMatrix(predictions, test_set$diabetes)

# Print confusion matrix details
print(confusion_matrix)

# Extract confusion matrix table
conf_matrix_table <- as.data.frame(confusion_matrix$table)

# Plot the confusion matrix using ggplot
ggplot(conf_matrix_table, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()


# Save the trained SVM model for future use
saveRDS(svm_model_tuned, file = "best_svm_model.rds")


