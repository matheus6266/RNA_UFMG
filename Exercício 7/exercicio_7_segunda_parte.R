rm(list = ls())
library(corpcor)
library(rgl)
library(mlbench)

source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 7\\treinaRBF.R")
source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 7\\YRBF.R")

# Gerar os dados
set.seed(123)  # Para reprodutibilidade
x <- runif(100, -15, 15)  # 100 valores entre -15 e 15
y <- sin(x) / x + rnorm(100, 0, 0.05)  # Função sinc(x) com ruído gaussiano

y[is.nan(y)] <- 1  # Definindo sinc(0) = 1

# Divisão dos dados em treino (70%) e validação (30%)
set.seed(456)
indices_treino <- sample(1:length(x), size = floor(0.7 * length(x)))
indices_validacao <- setdiff(1:length(x), indices_treino)

# Conjunto de treino
x_treino <- x[indices_treino]
y_treino <- y[indices_treino]

# Conjunto de validação
x_validacao <- x[indices_validacao]
y_validacao <- y[indices_validacao]

# Configurar parâmetros
p <- 20  # Número de centros
r <- 10   # Spread (ajustável)

# Treinar a Rede RBF
modRBF <- treinaRBF(as.matrix(x_treino), as.matrix(y_treino), p, r)

# Predições no conjunto de validação
y_pred <- YRBF(as.matrix(x_validacao), modRBF)

# Calcular o erro médio quadrático (RMSE)
rmse <- mean((y_validacao - y_pred)^2)

cat("RMSE do modelo:", round(rmse, 4), "\n")

# Função sinc(x) real
x_seq <- seq(-15, 15, length.out = 200)
y_real <- sin(x_seq) / x_seq
y_real[is.nan(y_real)] <- 1  # Para sinc(0)

# Predições do modelo
y_model <- YRBF(as.matrix(x_seq), modRBF)

# Plotar os dados reais e as predições
plot(x, y, col = "blue", pch = 19, main = paste("Aproximação da Função sinc(x) (p =", p, ")"),
     xlab = "x", ylab = "y")
lines(x_seq, y_real, col = "black", lwd = 2, lty = 2)  # Função sinc(x) real
lines(x_seq, y_model, col = "red", lwd = 2)  # Predição do modelo RBF
legend("topright", legend = c("Dados com Ruído", "Função sinc(x)", "Modelo RBF"),
       col = c("blue", "black", "red"), pch = c(19, NA, NA), lty = c(NA, 2, 1), lwd = c(NA, 2, 2))

############################## Teste com 50 Amostras


# Gerar os dados
set.seed(123)  # Para reprodutibilidade
x <- runif(50, -15, 15)  # 50 valores entre -15 e 15
y <- sin(x) / x + rnorm(50, 0, 0.05)  # Função sinc(x) com ruído gaussiano

y[is.nan(y)] <- 1  # Definindo sinc(0) = 1

# Divisão dos dados em treino (70%) e validação (30%)
set.seed(456)
indices_treino <- sample(1:length(x), size = floor(0.7 * length(x)))
indices_validacao <- setdiff(1:length(x), indices_treino)

# Conjunto de treino
x_treino <- x[indices_treino]
y_treino <- y[indices_treino]

# Conjunto de validação
x_validacao <- x[indices_validacao]
y_validacao <- y[indices_validacao]

# Configurar parâmetros
p <- 20  # Número de centros
r <- 10   # Spread (ajustável)

# Treinar a Rede RBF
modRBF <- treinaRBF(as.matrix(x_treino), as.matrix(y_treino), p, r)

# Predições no conjunto de validação
y_pred <- YRBF(as.matrix(x_validacao), modRBF)

# Calcular o erro médio quadrático (RMSE)
rmse <- mean((y_validacao - y_pred)^2)

cat("RMSE do modelo:", round(rmse, 4), "\n")

# Função sinc(x) real
x_seq <- seq(-15, 15, length.out = 200)
y_real <- sin(x_seq) / x_seq
y_real[is.nan(y_real)] <- 1  # Para sinc(0)

# Predições do modelo
y_model <- YRBF(as.matrix(x_seq), modRBF)

# Plotar os dados reais e as predições
plot(x, y, col = "blue", pch = 19, main = paste("Aproximação da Função sinc(x) (p =", p, ")"),
     xlab = "x", ylab = "y")
lines(x_seq, y_real, col = "black", lwd = 2, lty = 2)  # Função sinc(x) real
lines(x_seq, y_model, col = "red", lwd = 2)  # Predição do modelo RBF
legend("topright", legend = c("Dados com Ruído", "Função sinc(x)", "Modelo RBF"),
       col = c("blue", "black", "red"), pch = c(19, NA, NA), lty = c(NA, 2, 1), lwd = c(NA, 2, 2))
