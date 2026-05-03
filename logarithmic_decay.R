options(scipen = 100, digits = 4)

# Load necessary libraries
library(nls2)
library(readr)

#### Benchmarking Project data ####

# Example data
retention_data <- data.frame(
  Month = c(3, 6, 12), # Representing months since start
  Retention = c(0.770650822, 0.643844392, 0.502495769) # 2023 data
)

# Non-linear least squares fitting of exponential decay
model <- nls(Retention ~ a - b * log(Month),
             data = retention_data,
             start = list(a = 1, b = 0.05))  # Initial guesses for a and b

# Summary of the model to see the best fit decay rate
model_summary <- summary(model)

# For loop
months <- seq(1, 121, 1) # 10 years
# a <- model_summary$coefficients[1]
b <- model_summary$coefficients[2]
a <- 0.95 # Manually adjusting a
# b <- 0.145 # Manually adjusting b

# Initialise a list
ret_list = list()

for(month in months){
  
  r <- a - b * log(month)
  
  ret_list <- append(ret_list, r)
  
}

# Initialise the data frame
retention_rates <- data.frame(
  Month = months,
  stringsAsFactors = FALSE
)

# Add the list as a column
retention_rates$Retention <- ret_list

retention_rates$Retention <- as.numeric(retention_rates$Retention)

readr::write_csv(retention_rates, "C:/Users/kay.evon/OneDrive - Heart Research Institute/Regular Giving/10-year Projections/Retention Rates AQ 3.csv")

#### Upgrades ####

# Example data
retention_data_up <- data.frame(
  Month = c(3, 6, 12), # Representing months since start
  Retention = c(0.770650822, 0.643844392, 0.502495769) # 2023 data
)

# Non-linear least squares fitting of exponential decay
model <- nls(Retention ~ a - b * log(Month),
             data = retention_data,
             start = list(a = 1, b = 0.05))  # Initial guesses for a and b

# Summary of the model to see the best fit decay rate
model_summary <- summary(model)

# For loop
months <- seq(1, 121, 1) # 10 years
# a <- model_summary$coefficients[1]
b <- model_summary$coefficients[2]
a <- 0.95 # Manually adjusting a
# b <- 0.145 # Manually adjusting b

# Initialise a list
ret_list = list()

for(month in months){
  
  r <- a - b * log(month)
  
  ret_list <- append(ret_list, r)
  
}

# Initialise the data frame
retention_rates <- data.frame(
  Month = months,
  stringsAsFactors = FALSE
)

# Add the list as a column
retention_rates$Retention <- ret_list

retention_rates$Retention <- as.numeric(retention_rates$Retention)

readr::write_csv(retention_rates, "C:/Users/kay.evon/OneDrive - Heart Research Institute/Regular Giving/10-year Projections/Retention Rates AQ 3.csv")

#### Reactivation ####

# Example data
retention_data_ra <- data.frame(
  Month = c(1:66), # Representing months since start
  Retention = c(0.92, 	0.89, 	0.84, 	0.82, 	0.79, 	0.754, 	0.72, 	0.7, 	0.68, 	0.65, 	0.62, 	0.593, 	0.57, 	0.55, 	0.54, 	0.53, 	0.51, 	0.49, 	0.48, 	0.47, 	0.46, 	0.4558, 	0.44, 	0.439, 	0.42, 	0.41, 	0.4, 	0.39, 	0.38, 	0.37, 	0.36, 	0.35, 	0.34, 	0.3375, 	0.335, 	0.331, 	0.321, 	0.318, 	0.315, 	0.3128, 	0.31, 	0.3067, 	0.305, 	0.3025, 	0.3015, 	0.2985, 	0.2925, 	0.29, 	0.2869, 	0.27, 	0.26, 	0.25, 	0.24, 	0.23, 	0.22, 	0.21, 	0.2, 	0.1975, 	0.19, 	0.187, 	0.0749799430999006, 	0.0721947447686101, 	0.0695130051680669, 	0.066930881229414, 	0.0644446726380899, 	0.0620508165310904)
)

# Non-linear least squares fitting of exponential decay
model <- nls(Retention ~ a - b * log(Month),
             data = retention_data,
             start = list(a = 1, b = 0.05))  # Initial guesses for a and b

# Summary of the model to see the best fit decay rate
model_summary <- summary(model)

# For loop
months <- seq(1, 121, 1) # 10 years
a <- model_summary$coefficients[1]
b <- model_summary$coefficients[2]
# a <- 0.95 # Manually adjusting a
# b <- 0.145 # Manually adjusting b

# Initialise a list
ret_list = list()

for(month in months){
  
  r <- a - b * log(month)
  
  ret_list <- append(ret_list, r)
  
}

# Initialise the data frame
retention_rates <- data.frame(
  Month = months,
  stringsAsFactors = FALSE
)

# Add the list as a column
retention_rates$Retention <- ret_list

retention_rates$Retention <- as.numeric(retention_rates$Retention)

readr::write_csv(retention_rates, "C:/Users/kay.evon/OneDrive - Heart Research Institute/Regular Giving/10-year Projections/Retention Rates AQ 4.csv")
