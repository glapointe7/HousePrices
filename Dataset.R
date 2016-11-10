getUniqueValues <- function(feature)
{
    feature.values <- unique(feature)
    if(length(feature.values) <= 25)
    {
        paste(sort(feature.values, na.last = TRUE), collapse = ", ")
    }
}

getCategoryMean <- function(feature)
{
    values <- sort(unique(feature))
    values.num <- as.numeric(factor(feature))
    values.mean <- round(mean(values.num, na.rm = TRUE), 0)
    value <- values[values.mean]
    
    return(value)
}

printRMSEInformation <- function(train.prediction, train.sale.price)
{
    sale.price.log <- log(train.sale.price + 1)
    rmse <- RMSE(sale.price.log, train.prediction)
    cat("RMSE =", rmse)
    
    data.frame(predicted = exp(train.prediction) - 1, observed = train.sale.price) %>%
        ggplot(aes(x = predicted, y = observed)) +
        geom_point() +
        geom_abline(slope = 1, intercept = 0) +
        labs(title = "Predicted sale price in function of the observed sale price",
             x = 'Observed sale price ($)', 
             y = 'Predicted sale price ($)')
    
    return(rmse)
}