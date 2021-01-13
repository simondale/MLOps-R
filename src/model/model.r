library(caret)
library(optparse)

train_model <- function(args) {
    options <- list(
        optparse::make_option("--model-output")
    )

    parser <- optparse::OptionParser(option_list = options)
    opts <- optparse::parse_args(
        parser,
        args = args,
        convert_hyphens_to_underscores = TRUE
    )

    print("Training model")
    model <- caret::train(Species ~ ., data = iris, method = "rpart")

    print(paste("Saving model to", opts$model_output))
    saveRDS(model, opts$model_output)
    print("Model saved")
}