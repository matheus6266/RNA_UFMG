rm(list = ls())
library(corpcor)
library(rgl)
library(mlbench)
library(rgl)


source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 7\\treinaRBF.R")
source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 7\\YRBF.R")

data_2dnormals <- mlbench.2dnormals(200)


# Suponha que seus dados importados estejam no formato de lista
# com 'x' representando as características e 'classes' representando as saídas
dados <- data_2dnormals  # Substitua por qualquer conjunto de dados importado
xin <- dados$x
yin <- as.numeric(dados$classes)

# Dividir os dados em 70% treino e 30% validação
#set.seed(178215)  # Para reprodutibilidade
n_total <- nrow(xin)
indices_treino <- sample(1:n_total, size = floor(0.7 * n_total))
indices_validacao <- setdiff(1:n_total, indices_treino)

# Conjunto de treino
xin_treino <- xin[indices_treino, ]
yin_treino <- yin[indices_treino]



# Conjunto de validação
xin_validacao <- xin[indices_validacao, ]
yin_validacao <- yin[indices_validacao]

yin_treino <- ifelse(yin_treino == 1, -1, 1)
yin_validacao <- ifelse(yin_validacao == 1, -1, 1)

# Plotar os dados de treino com as cores ajustadas
plot(xin_treino[yin_treino == -1, 1], xin_treino[yin_treino == -1, 2], col = "red", pch = 19,
     xlim = range(xin_treino[, 1]), ylim = range(xin_treino[, 2]),
     main = "Estrutura dos Dados de Treino", xlab = "X1", ylab = "X2")
points(xin_treino[yin_treino == 1, 1], xin_treino[yin_treino == 1, 2], col = "blue", pch = 19)

# Plotar os dados de validação com as cores ajustadas
plot(xin_validacao[yin_validacao == -1, 1], xin_validacao[yin_validacao == -1, 2], col = "red", pch = 19,
     xlim = range(xin_validacao[, 1]), ylim = range(xin_validacao[, 2]),
     main = "Estrutura dos Dados de Validação", xlab = "X1", ylab = "X2")
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



