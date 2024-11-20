rm(list = ls())
library(corpcor)
library(rgl)
library(mlbench)
library(rgl)
library(Rtsne)
library(ggplot2)
library(caret)


# Função para normalizar os dados
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 6\\statlog\\trainELM.R")
source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 6\\statlog\\YELM.R")


data_file <- "C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 6\\statlog\\statlog+heart\\heart.dat"

data <- read.table(data_file, sep = "", header = FALSE)


# Converter Diagnosis para -1 e 1
data$V14 <- ifelse(data$V14 == 1, 1, -1)
data$V14 <- as.factor(data$V14)


# Executar t-SNE
set.seed(123)  # Para reprodutibilidade
tsne_results <- Rtsne(data[, -1], dims = 2, perplexity = 30, verbose = TRUE, max_iter = 1000)

# Criar um data frame com os resultados do t-SNE
tsne_data <- data.frame(X = tsne_results$Y[,1], Y = tsne_results$Y[,2], Diagnosis = data$V14)

# Plotar os resultados
ggplot(tsne_data, aes(x = X, y = Y, color = Diagnosis)) +
  geom_point() +
  labs(title = "t-SNE Plot of Statlog", x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
  theme_minimal()


# Criar índices para dividir o conjunto de dados t-SNE em 70% para treinamento e 30% para teste
train_index <- createDataPartition(tsne_data$Diagnosis, p = 0.7, list = FALSE)

# Dividir o conjunto de dados em treino e teste
tsne_train_data <- tsne_data[train_index, ]
tsne_test_data <- tsne_data[-train_index, ]



# Aplicar a normalização aos atributos X e Y dos conjuntos de treino e teste
tsne_train_data$X <- normalize(tsne_train_data$X)
tsne_train_data$Y <- normalize(tsne_train_data$Y)

tsne_test_data$X <- normalize(tsne_test_data$X)
tsne_test_data$Y <- normalize(tsne_test_data$Y)


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


# Preparar os dados de entrada (features) e saída (labels) do conjunto de treinamento
xin <- as.matrix(tsne_train_data[, c("X", "Y")])  # Selecionar as colunas X e Y como entrada
yin <- as.matrix(tsne_train_data$Diagnosis)       # Selecionar a coluna Diagnosis como saída

# Converter 'yin' para numérico se necessário
yin <- as.numeric(as.character(tsne_train_data$Diagnosis))

# Definir o número de neurônios na camada oculta (p) e o parâmetro 'par'
p <- 1500  # Exemplo: número de neurônios, ajuste conforme necessário
par <- 1 # Adicionar bias

numero_testes <- 50
# Inicializar um vetor para armazenar as acurácias das 10 execuções
acuracias_treino <- numeric(numero_testes)
# Preparar os dados de entrada (features) do conjunto de treinamento
xin_treino <- as.matrix(tsne_train_data[, c("X", "Y")])
yin_treino <- as.numeric(as.character(tsne_train_data$Diagnosis))



# Executar o treinamento e teste 10 vezes
for (i in 1:numero_testes) {
  
  resultadoELM <- treinaELM(xin_treino, yin_treino, p = p, par = 1)  # Ajuste 'p' conforme necessário
  W <- resultadoELM[[1]]
  Z <- resultadoELM[[3]]
  # Fazer as previsões no conjunto de treinamento
  y_pred_treino <- YELM(xin_treino, Z, W, par = 1)
  
  # Calcular a acurácia e armazenar no vetor
  acuracias_treino[i] <- mean(y_pred_treino == yin_treino)
}

# Calcular a média e o desvio padrão das acurácias
media_treino <- mean(acuracias_treino)
desvio_padrao_treino <- sd(acuracias_treino)

# Apresentar a acurácia na forma 'média ± desvio_padrão'
acuracia_resultado_treino <- sprintf("%.4f ± %.4f", media_treino, desvio_padrao_treino)
print(acuracia_resultado_treino)


seqx1x2 <- seq(0, 1, 0.1)
lseq <- length(seqx1x2)
MZ <- matrix(nrow = lseq, ncol = lseq)

# Calcular a superfície de separação usando loops
for (i in 1:lseq) {
  for (j in 1:lseq) {
    x1 <- seqx1x2[i]
    x2 <- seqx1x2[j]
    x1x2 <- as.matrix(cbind(1, x1, x2))  # Adiciona o bias
    h1 <- cbind(1, tanh(x1x2 %*% Z))      # Calcula a ativação na camada oculta
    MZ[i, j] <- sign(h1 %*% W)            # Calcula a previsão
  }
}

main_title <- paste("Superfície de Separação Conjunto de Treinamento p=", p)

# Plotar a superfície de separação
contour(seqx1x2, seqx1x2, MZ, nlevels = 1,
        xlim = range(seqx1x2), ylim = range(seqx1x2), 
        xlab = "X1", ylab = "X2", 
        main = main_title, 
        drawlabels = FALSE, col = "black")

# Separar os pontos de validação por classe
class1_points_treino <- xin_treino[yin_treino == 1, ]
class2_points_treino <- xin_treino[yin_treino == -1, ]

# Adicionar os pontos de validação ao gráfico
points(class1_points_treino[, 1], class1_points_treino[, 2], col = "red", pch = 19)
points(class2_points_treino[, 1], class2_points_treino[, 2], col = "blue", pch = 19)

#############################Conjunto Validação#################################################

# Preparar os dados de entrada e saída para validação (conjunto de teste)
xin_validacao <- as.matrix(tsne_test_data[, c("X", "Y")])
yin_validacao <- as.numeric(as.character(tsne_test_data$Diagnosis))

# Inicializar um vetor para armazenar as acurácias das 10 execuções no conjunto de validação
acuracias_validacao <- numeric(numero_testes)



# Executar o treinamento e teste 10 vezes
for (i in 1:numero_testes) {
  
  resultadoELM <- treinaELM(xin_treino, yin_treino, p = p, par = 1)  # Ajuste 'p' conforme necessário
  W <- resultadoELM[[1]]
  Z <- resultadoELM[[3]]
  # Fazer as previsões no conjunto de validação
  y_pred_validacao <- YELM(xin_validacao, Z, W, par = 1)
  
  # Calcular a acurácia e armazenar no vetor
  acuracias_validacao[i] <- mean(y_pred_validacao == yin_validacao)
}

# Calcular a média e o desvio padrão das acurácias
media_validacao <- mean(acuracias_validacao)
desvio_padrao_validacao <- sd(acuracias_validacao)

# Apresentar a acurácia na forma 'média ± desvio_padrão'
acuracia_resultado_validacao <- sprintf("%.4f ± %.4f", media_validacao, desvio_padrao_validacao)
print(acuracia_resultado_validacao)



seqx1x2 <- seq(0, 1, 0.1)
lseq <- length(seqx1x2)
MZ <- matrix(nrow = lseq, ncol = lseq)

# Calcular a superfície de separação usando loops
for (i in 1:lseq) {
  for (j in 1:lseq) {
    x1 <- seqx1x2[i]
    x2 <- seqx1x2[j]
    x1x2 <- as.matrix(cbind(1, x1, x2))  # Adiciona o bias
    h1 <- cbind(1, tanh(x1x2 %*% Z))      # Calcula a ativação na camada oculta
    MZ[i, j] <- sign(h1 %*% W)            # Calcula a previsão
  }
}

main_title <- paste("Superfície de Separação Conjunto do Validação p=", p)

# Plotar a superfície de separação

contour(seqx1x2, seqx1x2, MZ, nlevels = 1,
        xlim = range(seqx1x2), ylim = range(seqx1x2), 
        xlab = "X1", ylab = "X2", 
        main = main_title, 
        drawlabels = FALSE, col = "black")

# Separar os pontos de validação por classe
class1_points <- xin_validacao[yin_validacao == 1, ]
class2_points <- xin_validacao[yin_validacao == -1, ]

# Adicionar os pontos de validação ao gráfico
points(class1_points[, 1], class1_points[, 2], col = "red", pch = 19)
points(class2_points[, 1], class2_points[, 2], col = "blue", pch = 19)


# Carregar o pacote rgl


# Plotar a superfície de separação em 3D
persp3d(seqx1x2, seqx1x2, MZ, col = "red", 
        xlim = range(seqx1x2), ylim = range(seqx1x2), 
        zlim = c(min(MZ), max(MZ)),  # Definindo os limites do eixo Z
        xlab = "X1", ylab = "X2", zlab = "Classe")
