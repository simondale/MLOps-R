library(caret)

model <- caret::train(Species ~ ., data = iris, method = "rpart")

saveRDS(model, "model.rds")
