# Load the train and test datasets
train_set <- readRDS("train_set.rds")
test_set <- readRDS("test_set.rds")

# Fit the logistic regression model
logistic_model <- glm(diabetes ~ ., data = train_set, family = binomial)

# Print the summary of the model
summary(logistic_model)
# Predict probabilities on the test set using the logistic model
logistic_pred_prob <- predict(logistic_model, newdata = test_set, type = "response")

# Classify as 'pos' if probability >= 0.5, otherwise 'neg'
logistic_pred_class <- ifelse(logistic_pred_prob >= 0.5, "pos", "neg")

# Compare predictions with the actual values (test_set$diabetes)
accuracy_logistic <- mean(logistic_pred_class == test_set$diabetes)

# Print the accuracy
cat("Logistic Regression Model Accuracy:", accuracy_logistic, "\n")

# Load the glmnet package
library(glmnet)

# Prepare the training data for glmnet
x_train <- as.matrix(train_set[, -which(names(train_set) == "diabetes")])  # All columns except 'diabetes'
y_train <- train_set$diabetes  # Target variable

# Perform cross-validation with Lasso (L1 regularization) and Ridge (L2 regularization)
cv_logistic <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")  # Lasso (alpha = 1)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial")  # Ridge (alpha = 0)

# Plot the cross-validation results for Lasso and Ridge
plot(cv_logistic)
plot(cv_ridge)

# Best lambda for Lasso and Ridge
best_lambda_lasso <- cv_logistic$lambda.min
best_lambda_ridge <- cv_ridge$lambda.min

# Display the best lambda values
cat("Best lambda for Lasso (L1):", best_lambda_lasso, "\n")
cat("Best lambda for Ridge (L2):", best_lambda_ridge, "\n")


# Fit the Lasso model using the best lambda
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda_lasso, family = "binomial")

# Fit the Ridge model using the best lambda
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda_ridge, family = "binomial")

# Print the coefficients of each model
cat("Lasso Model Coefficients:\n")
print(coef(lasso_model))

cat("\nRidge Model Coefficients:\n")
print(coef(ridge_model))


# Prepare the test data for prediction
x_test <- as.matrix(test_set[, -which(names(test_set) == "diabetes")])
y_test <- test_set$diabetes

# Predict using the Lasso model

lasso_pred <- predict(lasso_model, s = best_lambda_lasso, newx = x_test, type = "response")
lasso_class <- ifelse(lasso_pred >= 0.5, "pos", "neg")

# Predict using the Ridge model
ridge_pred <- predict(ridge_model, s = best_lambda_ridge, newx = x_test, type = "response")
ridge_class <- ifelse(ridge_pred >= 0.5, "pos", "neg")

# Evaluate accuracy for Lasso and Ridge
lasso_accuracy <- mean(lasso_class == y_test)
ridge_accuracy <- mean(ridge_class == y_test)

cat("Lasso Model Accuracy:", lasso_accuracy, "\n")
cat("Ridge Model Accuracy:", ridge_accuracy, "\n")


# Load necessary libraries
library(caret)
library(ggplot2)
library(tidyr)

# Create the confusion matrix
conf_matrix_logistic <- confusionMatrix(as.factor(logistic_pred_class), as.factor(test_set$diabetes))

# Convert the confusion matrix into a tidy data frame
conf_matrix_df <- as.data.frame(conf_matrix_logistic$table)

# Plot the confusion matrix using ggplot2
ggplot(conf_matrix_df, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +  # Create the tiles
  geom_text(aes(label = Freq), vjust = 1) +  # Add the count values on top of each tile
  scale_fill_gradient(low = "white", high = "red") +  # Set color gradient
  theme_minimal() +  # Clean theme
  labs(
       x = "Actual Class",
       y = "Predicted Class",
       fill = "Frequency") +  # Labels
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels


# Save the logistic regression model
save(logistic_model, file = "logistic_model.RData")
