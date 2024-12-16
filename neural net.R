set.seed(123)
library(neuralnet)

# Load train and test sets from .rds files
train_set <- readRDS("train_set.rds") 
test_set <- readRDS("test_set.rds")    

# Convert target variable 'diabetes' to binary values
train_set$diabetes <- ifelse(train_set$diabetes == "pos", 1, 0)
test_set$diabetes <- ifelse(test_set$diabetes == "pos", 1, 0)

# Check if the conversion was successful
table(train_set$diabetes)
table(test_set$diabetes)

# Fit a neural network model using the neuralnet package
library(neuralnet)

nn_model <- neuralnet(
  diabetes ~ .,                       
  data = train_set,                   
  hidden = c(3, 3),                  
  linear.output = FALSE,              # Classification problem (binary output)
  lifesign = 'full',                  # Verbose output during training
  rep = 1                             # Number of repetitions
)

# Plot the trained neural network model
plot(nn_model)


# Use the model to make predictions on the test set
predictions <- compute(nn_model, test_set[, -ncol(test_set)])  # Exclude the target column

# Convert the predictions to binary output (0 or 1)
predicted_class <- ifelse(predictions$net.result > 0.5, 1, 0)

# Evaluate model performance (e.g., confusion matrix)
library(caret)
conf_matrix <- confusionMatrix(as.factor(predicted_class), as.factor(test_set$diabetes))
print(conf_matrix)


##antoher nn with 8, 4 node respectively
nn_model <- neuralnet(
  diabetes ~ .,                       
  data = train_set,                   # Training data
  hidden = c(8, 4),                  
  linear.output = FALSE,              # Classification problem (binary output)
  lifesign = 'full',                  # Verbose output during training
  rep = 1                             # Number of repetitions
)

# Plot the trained neural network model
plot(nn_model)


# Use the model to make predictions on the test set
predictions <- compute(nn_model, test_set[, -ncol(test_set)])  # Exclude the target column

# Convert the predictions to binary output (0 or 1)
predicted_class <- ifelse(predictions$net.result > 0.5, 1, 0)

# Evaluate model performance (e.g., confusion matrix)
library(caret)
conf_matrix <- confusionMatrix(as.factor(predicted_class), as.factor(test_set$diabetes))
print(conf_matrix)


# Save the trained neural network model to an .rds file
saveRDS(nn_model, file = "nn_model.rds")


