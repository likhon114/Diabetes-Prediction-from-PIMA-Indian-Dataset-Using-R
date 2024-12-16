# Install the package
install.packages("mlbench")

# Load the required libraries
library(mlbench)
library(dplyr)

# Load the Pima Indians Diabetes dataset
data(PimaIndiansDiabetes)
df <- PimaIndiansDiabetes

# View the first few rows
head(df)
str(df)
summary(df)

# Check for missing values
sum(is.na(df))

# Alternatively, use sapply to check for missing values column-wise
sapply(df, function(x) sum(is.na(x)))

library(ggplot2)

# Plot the target variable
ggplot(df, aes(x = diabetes)) +
  geom_bar(fill = "lightblue") +
  labs(title = "Distribution of Diabetes Outcome")

library(corrplot)

# Plot a correlation matrix for numerical variables
cor_matrix <- cor(df[,-9]) # Exclude the diabetes column as it's categorical
corrplot(cor_matrix, method = "circle", tl.cex = 0.8)

# Boxplot of glucose levels by diabetes outcome
ggplot(df, aes(x = diabetes, y = glucose)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Glucose Levels by Diabetes Outcome")

# Check class distribution of diabetes outcome
table(df$diabetes)

# Proportion of each class
prop.table(table(df$diabetes))

# Check for zero values in columns where 0 might not make sense
sapply(df[, 1:8], function(x) sum(x == 0))

# Replace zero values with the median for appropriate columns including triceps
df$glucose[df$glucose == 0] <- median(df$glucose[df$glucose != 0], na.rm = TRUE)
df$pressure[df$pressure == 0] <- median(df$pressure[df$pressure != 0], na.rm = TRUE)
df$triceps[df$triceps == 0] <- median(df$triceps[df$triceps != 0], na.rm = TRUE) # For triceps
df$insulin[df$insulin == 0] <- median(df$insulin[df$insulin != 0], na.rm = TRUE)
df$mass[df$mass == 0] <- median(df$mass[df$mass != 0], na.rm = TRUE)



# Check for zero values in columns where 0 might not make sense
sapply(df[, 1:8], function(x) sum(x == 0))



# Scale the numerical columns (excluding the target variable "diabetes")
df_scaled <- df
df_scaled[, 1:8] <- scale(df[, 1:8])
# Check summary statistics to confirm changes
summary(df_scaled)


############starting with differnt model################################


########split test-train##############


# Set seed for reproducibility
set.seed(123)

# Create a random sample of row indices (70% for training)
train_indices <- sample(1:nrow(df_scaled), size = 0.7 * nrow(df_scaled))

# Split the data into training and testing sets
train_set <- df_scaled[train_indices, ]
test_set <- df_scaled[-train_indices, ]

########split test-train##############



# Save the datasets
saveRDS(train_set, file = "train_set.rds")
saveRDS(test_set, file = "test_set.rds")

