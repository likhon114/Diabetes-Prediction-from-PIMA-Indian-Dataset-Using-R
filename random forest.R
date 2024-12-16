set.seed(123)
# Load the train and test set
train_set <- readRDS("train_set.rds")
test_set <- readRDS("test_set.rds")

# Load necessary library
library(randomForest)
library(caret)

# Set up 10-fold cross-validation
control <- trainControl(method = "cv", number = 10)

tune_grid <- expand.grid(mtry = c(2, 3, 4))

rf_model_cv <- train(
  diabetes ~ ., 
  data = train_set, 
  method = "rf", 
  trControl = control,
  tuneGrid = tune_grid  )

# Print the results of the cross-validation
print(rf_model_cv)


# Evaluate the accuracy on the test set
pred_rf <- predict(rf_model_cv, newdata = test_set)
accuracy_rf <- sum(pred_rf == test_set$diabetes) / nrow(test_set)
cat("Accuracy of the Random Forest model: ", accuracy_rf, "\n")


# Set a random seed for reproducibility
set.seed(123)


library(ggplot2)  # For plotting

# Set different values for ntree (1 to 2000)
ntree_values <- seq(1, 2000, by = 100) 
# Create a tuning grid with mtry set to 3
tune_grid <- expand.grid(mtry = 3)

# Initialize empty lists to store results
accuracy_results <- data.frame(ntree = integer(), Train_Accuracy = numeric(), Test_Accuracy = numeric())

for (ntree in ntree_values) {
  # Train the Random Forest model for each ntree
  rf_model <- train(
    diabetes ~ ., 
    data = train_set, 
    method = "rf", 
    trControl = control, 
    tuneGrid = tune_grid,  
    ntree = ntree    # Set ntree manually
  )
  
  # Calculate accuracy on the training set
  train_accuracy <- max(rf_model$results$Accuracy)
  
  # Predict on the test set
  test_predictions <- predict(rf_model, newdata = test_set)
  
  # Calculate accuracy on the test set
  test_accuracy <- sum(test_predictions == test_set$diabetes) / nrow(test_set)
  
  # Save the ntree value, training accuracy, and test accuracy
  accuracy_results <- rbind(accuracy_results, data.frame(ntree = ntree, Train_Accuracy = train_accuracy, Test_Accuracy = test_accuracy))
}

# Print the accuracy results
print(accuracy_results)

# Plot accuracy against ntree values for both train and test sets
library(ggplot2)
ggplot(accuracy_results, aes(x = ntree)) +
  geom_line(aes(y = Train_Accuracy, color = "Train"), size = 1) + 
  geom_point(aes(y = Train_Accuracy, color = "Train"), size = 2) +
  geom_line(aes(y = Test_Accuracy, color = "Test"), size = 1, linetype = "dashed") +
  geom_point(aes(y = Test_Accuracy, color = "Test"), size = 2) +
  labs(title = "Random Forest Accuracy vs. Number of Trees (ntree)",
       x = "Number of Trees (ntree)",
       y = "Accuracy") +
  scale_color_manual(values = c("Train" = "blue", "Test" = "red")) +
  theme_minimal()

# Find the row with the best accuracy on the test set
best_result <- accuracy_results[which.max(accuracy_results$Test_Accuracy), ]

# Print the ntree value and corresponding test accuracy
print(best_result)



# Set the parameters
mtry_value <- 3
ntree_value <- 500

# Train the Random Forest model with mtry = 3 and ntree = 500
rf_model_final <- train(
  diabetes ~ ., 
  data = train_set, 
  method = "rf", 
  trControl = control, 
  tuneGrid = expand.grid(mtry = mtry_value),  # Use fixed mtry value
  ntree = ntree_value    # Set ntree to 500
)

# Print the model details
print(rf_model_final)

# Save the trained model for future use
save(rf_model_final, file = "rf_model_final.RData")



# Extract feature importance from the random forest model
feature_importance <- importance(rf_model_final$finalModel)

# Print the feature importance
print(feature_importance)

# Create a data frame for visualization
feature_importance_df <- data.frame(
  Feature = rownames(feature_importance),
  Importance = feature_importance[, 1]
)

# Sort by importance
feature_importance_df <- feature_importance_df[order(-feature_importance_df$Importance), ]

# Plot the importance of each feature
ggplot(feature_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Feature Importance in Random Forest Model",
       x = "Feature",
       y = "Importance") +
  theme_minimal()