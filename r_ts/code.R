# Load required packages
library(readxl)
library(tidyverse)
library(lubridate)
library(ggplot2)
library(cowplot)
library(lmtest)
library(urca)
library(psych)
library(vars)
library(dplyr)
library(forecast)

# import data
df <- read_excel("Data Yearly.xlsx")
# Convert Dates column to date type
df$Dates <- as.Date(df$Dates, format = "%Y-%m-%d")
# rename var names to shorter names for eacy visualization
df2 <- rename(df, 
              Dates = Dates,
              TradeBalance = `Trade Balance (Million Dollar)`,
              CurrentAccount = `Balance of Payment Current Account (Million Dollar)`,
              HotMoney = `Hot Money (Million Dollar)`,
              GDP = `GDP Current Prices (Million Dollar)`,
              USDTRY = `USDTRY Curncy`,
              XU100 = `XU100 Index`
)
# check missing data
sum(is.na(df)) # no missing data, imputation not necessary
#df <- na.omit(df)
# summary statistics
describe(df2)


# Visualization 
# Exploring all variables with histogram
df2 %>%
  gather(key = "Variable", value = "Value", -Dates) %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ Variable, scales = "free") +
  theme_light()
# Plotting variables over time
df2 %>%
  gather(key = "Variable", value = "Value", -Dates) %>%
  ggplot(aes(x = Dates, y = Value)) +
  geom_line() +
  facet_wrap(~ Variable, scales = "free_y") +
  theme_classic()
# Plotting correlation matrix
df_cor <- cor(df2[, -1])
corrplot::corrplot(df_cor, method = "number")



# MODELLING USING LINEAR MODEL
# Model Building - estimating effects of hot money on stock exchange index, exchange rates, and interest rates.
model_XU100 <- lm(XU100 ~ HotMoney, data = df2)
model_USDTRY <- lm(USDTRY ~ HotMoney, data = df2)
summary(model_XU100)
summary(model_USDTRY)


# Testing for Granger causality
# verify the usefulness of one variable to forecast another.
gc_XU100 <- grangertest(XU100 ~ HotMoney, order = 1, data = df2)
summary(gc_XU100)
gc_USDTRY <- grangertest(USDTRY ~ HotMoney, order = 1, data = df2)
summary(gc_USDTRY)

# Create a new data frame for VAR model
df_vars <- df2[, c("Dates", "HotMoney", "XU100", "USDTRY")]
# Ensure the dataset is a time-series object
df_vars_ts <- ts(df_vars[-1], start = year(df_vars$Dates[1]), frequency = 1)
# Done
# Perform the Augmented Dickey-Fuller test for stationarity
adf_HotMoney <- ur.df(df_vars$HotMoney, type = "drift")
adf_XU100 <- ur.df(df_vars$XU100, type = "drift")
adf_USDTRY <- ur.df(df_vars$USDTRY, type = "drift")
summary(adf_HotMoney)
summary(adf_XU100)
summary(adf_USDTRY)

# VAR MODELLING
var_model <- VAR(df_vars[-1], lag.max = 1, type = "const")
# Summary of the model
summary(var_model)
# Plot the impulse response functions to see how a shock to one variable affects the other variables
irf_res <- irf(var_model, impulse = "HotMoney", response = c("XU100", "USDTRY"), boot = TRUE, n.ahead = 10)
plot(irf_res)
# Perform Granger causality tests
causality(var_model, cause = "HotMoney")



# ARIMA MODELLING
# Splitting data into train and test sets
n <- nrow(df_vars)
train_frac <- 0.8 # adjust this as needed
n_train <- floor(train_frac * n)
n_test <- n - n_train

# Train data
train_data <- df_vars[1:n_train, ]

# Test data
test_data <- df_vars[(n_train+1):n, ]

# Fit the models on the train data
model_XU100_train <- auto.arima(train_data$XU100, xreg = train_data$HotMoney)
model_USDTRY_train <- auto.arima(train_data$USDTRY, xreg = train_data$HotMoney)

# Generate forecasts for the test period
forecast_XU100 <- forecast(model_XU100_train, h = n_test, xreg = test_data$HotMoney)
forecast_USDTRY <- forecast(model_USDTRY_train, h = n_test, xreg = test_data$HotMoney)
#plotting the forecast
plot(forecast_XU100)
plot(forecast_USDTRY)
# Performance measures
accuracy_XU100 <- accuracy(forecast_XU100, test_data$XU100)
accuracy_USDTRY <- accuracy(forecast_USDTRY, test_data$USDTRY)
# Output the performance measures
print(paste("XU100 model performance:"))
print(accuracy_XU100)
print(paste("USDTRY model performance:"))
print(accuracy_USDTRY)

# Note
# The performance measures include RMSE, MAE, MPE, MAPE and more.
# Lower values indicate better predictive accuracy.



