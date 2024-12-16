# Load the training and testing datasets
train_set <- readRDS("train_set.rds")
test_set <- readRDS("test_set.rds")

# Load necessary libraries
library(rpart)
library(caret)
library(rpart.plot)

# Set up control for 10-fold cross-validation
control <- trainControl(method = "cv", number = 10)

# Fit a simple decision tree with 10-fold CV
dt_model_cv <- train(
  diabetes ~ ., 
  data = train_set, 
  method = "rpart", 
  trControl = control,
  tuneLength = 1  )

# Print the summary of the decision tree model
print(dt_model_cv)
# Plot the general decision tree
rpart.plot(dt_model_cv$finalModel, main = "General Decision Tree", type = 3, extra = 101)

# Make predictions on the test set
predictions_dt <- predict(dt_model_cv, newdata = test_set)

# Confusion matrix and accuracy
conf_matrix_dt <- confusionMatrix(predictions_dt, test_set$diabetes)
print(conf_matrix_dt)


# Define the grid for tuning
tune_grid <- expand.grid(cp = seq(0.001, 0.1, by = 0.01))

# Fit a decision tree with 10-fold CV and tune the parameters
dt_model_tuned <- train(
  diabetes ~ ., 
  data = train_set, 
  method = "rpart", 
  trControl = control,
  tuneGrid = tune_grid  )

# Print the best model and its parameters
print(dt_model_tuned)


# Make predictions on the test set with the tuned model
predictions_dt_tuned <- predict(dt_model_tuned, newdata = test_set)

# Confusion matrix and accuracy for the tuned model
conf_matrix_dt_tuned <- confusionMatrix(predictions_dt_tuned, test_set$diabetes)
print(conf_matrix_dt_tuned)

# Plot the decision tree
rpart.plot(dt_model_tuned$finalModel, type = 3, extra = 101, fallen.leaves = TRUE, 
           main = "Tuned Decision Tree for Diabetes Prediction")

# Save the tuned decision tree model to a file
save(dt_model_tuned, file = "dt_model_tuned.RData")



