# Load necessary libraries
library(caret)
library(ggplot2)

# Load the train and test datasets saved as RDS files
train_set <- readRDS("train_set.rds")
test_set <- readRDS("test_set.rds")

# Define the range of k values 
k_values <- seq(1, 20, by = 1)

# Initialize an empty data frame to store accuracy results
accuracy_results <- data.frame(k = integer(), Accuracy = numeric())

# Loop through each value of k and train the KNN model
for (k in k_values) {
  # Train the KNN model with each k
  knn_model <- train(
    diabetes ~ .,          
    data = train_set,      
    method = "knn",        
    tuneGrid = expand.grid(k = k),  
    trControl = trainControl(method = "cv", number = 10)  # 10-fold cross-validation
  )
  
  # Store the accuracy for each k
  accuracy_results <- rbind(accuracy_results, data.frame(k = k, Accuracy = max(knn_model$results$Accuracy)))
}

# Print the accuracy results for each k value
print(accuracy_results)


# Plot accuracy vs. k values
ggplot(accuracy_results, aes(x = k, y = Accuracy)) +
  geom_line(color = "blue") + 
  geom_point(color = "red") +
  labs(title = "KNN Accuracy vs. Number of Neighbors (k)",
       x = "Number of Neighbors (k)",
       y = "Accuracy") +
  theme_minimal()


# Identify the best k value based on the accuracy
best_k <- accuracy_results[which.max(accuracy_results$Accuracy), "k"]
cat("Best k value: ", best_k, "\n")


# Train the final KNN model with the best k value
best_knn_model <- train(
  diabetes ~ ., 
  data = train_set, 
  method = "knn", 
  tuneGrid = expand.grid(k = best_k), 
  trControl = trainControl(method = "cv", number = 10)
)

# Predict on the test set using the best model
test_predictions <- predict(best_knn_model, newdata = test_set)

# Calculate accuracy on the test set
test_accuracy <- mean(test_predictions == test_set$diabetes)
cat("Test set accuracy: ", test_accuracy, "\n")

# Save the trained KNN model for future use
saveRDS(knn_model, file = "best_knn_model.rds")




