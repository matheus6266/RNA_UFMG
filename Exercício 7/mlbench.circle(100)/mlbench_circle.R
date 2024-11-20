rm(list = ls())
library(corpcor)
library(rgl)
library(mlbench)

source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 7\\treinaRBF.R")
source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 7\\YRBF.R")

# Gerar o conjunto de dados Circle com 100 pontos
data_circle <- mlbench.circle(100)

# Atribuir os dados às variáveis xin e yin
xin <- data_circle$x
yin <- as.numeric(data_circle$classes)

# Dividir os dados em 70% treino e 30% validação
#set.seed(123)  # Para reprodutibilidade
n_total <- nrow(xin)
indices_treino <- sample(1:n_total, size = floor(0.7 * n_total))
indices_validacao <- setdiff(1:n_total, indices_treino)

# Conjunto de treino
xin_treino <- xin[indices_treino, ]
yin_treino <- yin[indices_treino]

# Conjunto de validação
xin_validacao <- xin[indices_validacao, ]
yin_validacao <- yin[indices_validacao]

# Converter as classes de 1 e 2 para -1 e 1
yin_treino <- ifelse(yin_treino == 1, -1, 1)
yin_validacao <- ifelse(yin_validacao == 1, -1, 1)

# Plotar os dados de treino com as cores ajustadas
plot(xin_treino[yin_treino == -1, 1], xin_treino[yin_treino == -1, 2], col = "red", pch = 19,
     xlim = range(xin), ylim = range(xin),
     main = "Estrutura dos Dados de Treino (Circle)", xlab = "Feature 1", ylab = "Feature 2")
points(xin_treino[yin_treino == 1, 1], xin_treino[yin_treino == 1, 2], col = "blue", pch = 19)

# Plotar os dados de validação com as cores ajustadas
plot(xin_validacao[yin_validacao == -1, 1], xin_validacao[yin_validacao == -1, 2], col = "red", pch = 19,
     xlim = range(xin), ylim = range(xin),
     main = "Estrutura dos Dados de Validação (Circle)", xlab = "Feature 1", ylab = "Feature 2")
points(xin_validacao[yin_validacao == 1, 1], xin_validacao[yin_validacao == 1, 2], col = "blue", pch = 19)

r <- 10

p <-20

modRBF <- treinaRBF(xin_treino, yin_treino, p, r )
Yhat_tst <- YRBF(xin_validacao, modRBF)

# Converter para classe binária
Yhat_bin <- ifelse(Yhat_tst > 0, 1, -1)

# Calcular a acurácia
acuracia <- mean(Yhat_bin == yin_validacao)

# Exibir a acurácia
cat("Acurácia do modelo:", round(acuracia * 100, 2), "%\n")

plot(yin_validacao, type = 'b', col = 'red',
     xlim = c(0, 60), ylim = c(min(yin_validacao), max(yin_validacao) + 1),
     xlab = "Amostra", ylab = "ytst, Yhat_tst")

par(new = TRUE)

plot(Yhat_tst, type = 'l', col = 'blue',
     xlim = c(0, 60), ylim = c(min(yin_validacao), max(yin_validacao) + 1),
     xlab = "Amostra", ylab = "ytst, Yhat_tst")

calculaMargemSeparacao <- function(modRBF, xin, yin, resolucao = 100) {
  # Defina os limites do gráfico com base no intervalo dos dados de entrada
  x_min <- min(xin[, 1]) - 1
  x_max <- max(xin[, 1]) + 1
  y_min <- min(xin[, 2]) - 1
  y_max <- max(xin[, 2]) + 1
  
  # Crie uma grade de pontos no espaço de entrada
  x_seq <- seq(x_min, x_max, length.out = resolucao)
  y_seq <- seq(y_min, y_max, length.out = resolucao)
  grid <- expand.grid(x_seq, y_seq)
  
  # Calcule as previsões do modelo RBF para cada ponto da grade
  previsoes <- YRBF(as.matrix(grid), modRBF)
  
  # Converta as previsões para binário se necessário
  previsoes_bin <- ifelse(previsoes > 0, 1, -1)
  
  # Plote os dados e a margem de separação
  plot(xin, col = ifelse(yin == 1, 'blue', 'red'), pch = 19,
       xlab = "x1", ylab = "x2", main = paste("Superfície de Separação e Centros (p =", p, ")"))
  
  # Adicione os contornos da margem de separação
  contour(x_seq, y_seq, matrix(previsoes_bin, length(x_seq), length(y_seq)),
          levels = c(0), add = TRUE, drawlabels = FALSE, col = "black")
  
  # Plote os centros da RBF como círculos
  #points(modRBF$m, col = "black", pch = 1, cex = 5, lwd = 2)
  points(modRBF$m[, 1], modRBF$m[, 2], col = "black", pch = 4, cex = 2, lwd = 2)
}

# Chamada da função
calculaMargemSeparacao(modRBF, xin_treino, yin_treino)


