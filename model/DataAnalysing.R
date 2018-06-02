data <- read_csv('/Users/Future/Desktop/Spring 2018/Projects/Bioinformatics/Feature Selection/model/Results/CAE/Damavand/encoded_results_0.01_128.csv', col_names = c("MSE"))
data$MSE <- as.numeric(data$MSE)
hist(x = data$MSE, breaks = 30, col = 'red', xlab = 'MSE Loss', main = 'Distribution Plot (stddev = 0.01)')
plot(density(data$MSE))

