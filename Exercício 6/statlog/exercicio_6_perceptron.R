rm(list = ls())
library(corpcor)
library(rgl)
library(mlbench)
library(rgl)
library(Rtsne)
library(ggplot2)
library(caret)
library('plot3D')


# Função para normalizar os dados
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}


source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 6\\statlog\\trainperceptron.R")
source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 6\\statlog\\yperceptron.R")

data_file <- "C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 6\\statlog\\statlog+heart\\heart.dat"

data <- read.table(data_file, sep = "", header = FALSE)

data$V14 <- ifelse(data$V14 == 1, 1, 0)
data$V14 <- as.factor(data$V14)

# Executar t-SNE
set.seed(123)  # Para reprodutibilidade
tsne_results <- Rtsne(data[, -1], dims = 2, perplexity = 30, verbose = TRUE, max_iter = 1000)

# Criar um data frame com os resultados do t-SNE
tsne_data <- data.frame(X = tsne_results$Y[,1], Y = tsne_results$Y[,2], Diagnosis = data$V2)

# Plotar os resultados
ggplot(tsne_data, aes(x = X, y = Y, color = Diagnosis)) +
  geom_point() +
  labs(title = "t-SNE Plot of Breast Cancer Data", x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
  theme_minimal()


# Criar índices para dividir o conjunto de dados t-SNE em 70% para treinamento e 30% para teste
train_index <- createDataPartition(tsne_data$Diagnosis, p = 0.7, list = FALSE)

# Dividir o conjunto de dados em treino e teste
tsne_train_data <- tsne_data[train_index, ]
tsne_test_data <- tsne_data[-train_index, ]



# Aplicar a normalização aos atributos X e Y dos conjuntos de treino e teste
#tsne_train_data$X <- normalize(tsne_train_data$X)
#tsne_train_data$Y <- normalize(tsne_train_data$Y)

#tsne_test_data$X <- normalize(tsne_test_data$X)
#tsne_test_data$Y <- normalize(tsne_test_data$Y)

# Plotar o conjunto de treinamento
ggplot(tsne_train_data, aes(x = X, y = Y, color = as.factor(Diagnosis))) +
  geom_point() +
  labs(title = "t-SNE - Conjunto de Treinamento", x = "t-SNE Dimensão 1", y = "t-SNE Dimensão 2", color = "Diagnóstico") +
  theme_minimal()

# Plotar o conjunto de teste
ggplot(tsne_test_data, aes(x = X, y = Y, color = as.factor(Diagnosis))) +
  geom_point() +
  labs(title = "t-SNE - Conjunto de Teste", x = "t-SNE Dimensão 1", y = "t-SNE Dimensão 2", color = "Diagnóstico") +
  theme_minimal()


xin_treino <- as.matrix(tsne_train_data[, c("X", "Y")])
yin_treino <- as.numeric(as.character(tsne_train_data$Diagnosis))

xin_validacao <- as.matrix(tsne_test_data[, c("X", "Y")])
yin_validacao <- as.numeric(as.character(tsne_test_data$Diagnosis))

numero_testes <- 50
acuracias_treino <- numeric(numero_testes)


#################################### Conjunto Treinamento ######################
for (i in 1:numero_testes) {
  
  
  retlist<-trainperceptron(xin_treino,yin_treino,0.1,0.01,300,1)
  wt<-retlist[[1]]
  
  y_pred_validacao<-yperceptron(xin_treino,wt,1)
  
  # Calcular a acurácia e armazenar no vetor
  acuracias_treino[i] <- mean(y_pred_validacao == yin_treino)
}

# Calcular a média e o desvio padrão das acurácias
media_validacao <- mean(acuracias_treino)
desvio_padrao_validacao <- sd(acuracias_treino)

# Apresentar a acurácia na forma 'média ± desvio_padrão'
acuracia_resultado_treino <- sprintf("%.4f ± %.4f", media_validacao, desvio_padrao_validacao)
print(acuracia_resultado_treino)


# Preparar os dados de entrada (features) do conjunto de teste e os rótulos
xintest <- as.matrix(tsne_train_data[, c("X", "Y")])
ytest <- as.numeric(as.character(tsne_train_data$Diagnosis))  # Certificar-se de que ytest seja numérico

# Plotar os pontos das diferentes classes no conjunto de teste
plot(xintest[ytest == 0, 1], xintest[ytest == 0, 2], col = 'red', pch = 19,
     xlim = range(xintest[, 1]), ylim = range(xintest[, 2]), xlab = '', ylab = '', main = "Classificação Perceptron - Conjunto de Treinamento")
points(xintest[ytest == 1, 1], xintest[ytest == 1, 2], col = 'green', pch = 19)

# Gerar uma grade de pontos para plotar a superfície de separação
seqi <- seq(min(xintest[, 1]), max(xintest[, 1]), length.out = 100)
seqj <- seq(min(xintest[, 2]), max(xintest[, 2]), length.out = 100)
grid <- expand.grid(seqi, seqj)
grid_matrix <- as.matrix(grid)

par <- 1

# Adicionar o bias se necessário
if (par == 1) {
  grid_matrix <- cbind(1, grid_matrix)
}

# Calcular a saída do Perceptron para cada ponto na grade
z <- grid_matrix %*% wt
z <- matrix(z, length(seqi), length(seqj))

# Plotar a superfície de separação usando `contour`
contour(seqi, seqj, z, levels = 0, add = TRUE)

# Adicionar a legenda
legend("topright", legend = c("Classe 0", "Classe 1"),
       col = c("red", "green"),
       pch = c(19, 19),
       title = "Legenda",
       cex = 0.7)



#################################### Conjunto Validação ######################
for (i in 1:numero_testes) {
  
  
  retlist<-trainperceptron(xin_treino,yin_treino,0.1,0.01,300,1)
  wt<-retlist[[1]]
  
  y_pred_validacao<-yperceptron(xin_validacao,wt,1)
  
  # Calcular a acurácia e armazenar no vetor
  acuracias_treino[i] <- mean(y_pred_validacao == yin_validacao)
}

# Calcular a média e o desvio padrão das acurácias
media_validacao <- mean(acuracias_treino)
desvio_padrao_validacao <- sd(acuracias_treino)

# Apresentar a acurácia na forma 'média ± desvio_padrão'
acuracia_resultado_treino <- sprintf("%.4f ± %.4f", media_validacao, desvio_padrao_validacao)
print(acuracia_resultado_treino)


# Preparar os dados de entrada (features) do conjunto de teste e os rótulos
xintest <- as.matrix(tsne_test_data[, c("X", "Y")])
ytest <- as.numeric(as.character(tsne_test_data$Diagnosis))  # Certificar-se de que ytest seja numérico

# Plotar os pontos das diferentes classes no conjunto de teste
plot(xintest[ytest == 0, 1], xintest[ytest == 0, 2], col = 'red', pch = 19,
     xlim = range(xintest[, 1]), ylim = range(xintest[, 2]), xlab = '', ylab = '', main = "Classificação Perceptron - Conjunto de Validação")
points(xintest[ytest == 1, 1], xintest[ytest == 1, 2], col = 'green', pch = 19)

# Gerar uma grade de pontos para plotar a superfície de separação
seqi <- seq(min(xintest[, 1]), max(xintest[, 1]), length.out = 100)
seqj <- seq(min(xintest[, 2]), max(xintest[, 2]), length.out = 100)
grid <- expand.grid(seqi, seqj)
grid_matrix <- as.matrix(grid)

par <- 1

# Adicionar o bias se necessário
if (par == 1) {
  grid_matrix <- cbind(1, grid_matrix)
}

# Calcular a saída do Perceptron para cada ponto na grade
z <- grid_matrix %*% wt
z <- matrix(z, length(seqi), length(seqj))

# Plotar a superfície de separação usando `contour`
contour(seqi, seqj, z, levels = 0, add = TRUE)

# Adicionar a legenda
legend("topright", legend = c("Classe 0", "Classe 1"),
       col = c("red", "green"),
       pch = c(19, 19),
       title = "Legenda",
       cex = 0.7)
