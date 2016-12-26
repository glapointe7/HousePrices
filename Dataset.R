getUniqueValues <- function(feature)
{
    feature.values <- unique(feature)
    if(length(feature.values) <= 25)
    {
        paste(sort(feature.values, na.last = TRUE), collapse = ", ")
    }
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


Mode <- function(x, na.rm = FALSE) 
{
    if(na.rm)
    {
        x = x[!is.na(x)]
    }
    
    ux <- unique(x)
    return(ux[which.max(tabulate(match(x, ux)))])
}