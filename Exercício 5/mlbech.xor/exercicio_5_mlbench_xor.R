rm(list = ls())
library(corpcor)
library(rgl)
library(mlbench)

source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 5\\trainELM.R")
source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 5\\YELM.R")


# Gerar o conjunto de dados XOR com 100 pontos
data_xor <- mlbench.xor(100)

# Atribuir os dados às variáveis xin e yin
xin <- data_xor$x
yin <- as.numeric(data_xor$classes)

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
     main = "Estrutura dos Dados de Treino (XOR)", xlab = "Feature 1", ylab = "Feature 2")
points(xin_treino[yin_treino == 1, 1], xin_treino[yin_treino == 1, 2], col = "blue", pch = 19)

# Plotar os dados de validação com as cores ajustadas
plot(xin_validacao[yin_validacao == -1, 1], xin_validacao[yin_validacao == -1, 2], col = "red", pch = 19,
     xlim = range(xin), ylim = range(xin),
     main = "Estrutura dos Dados de Validação (XOR)", xlab = "Feature 1", ylab = "Feature 2")
points(xin_validacao[yin_validacao == 1, 1], xin_validacao[yin_validacao == 1, 2], col = "blue", pch = 19)

# Treinar a ELM usando a função treinaELM (assumindo que você já tem essa função)
p <- 5  # Número de neurônios na camada oculta
par <- 1  # Adicionar bias

modelo_elm <- treinaELM(xin_treino, yin_treino, p, par)

# Acessar os componentes do modelo treinado
W <- modelo_elm[[1]]
H <- modelo_elm[[2]]
Z <- modelo_elm[[3]]

# Fazer previsões no conjunto de validação
predicoes <- YELM(xin_validacao, Z, W, par)

# Calcular a acurácia do modelo
acuracia <- mean(predicoes == yin_validacao)
print(paste("Acurácia: ", round(acuracia * 100, 2), "%"))

# Plotar a superfície de separação
seqx1x2 <- seq(min(xin) - 1, max(xin) + 1, 0.1)
lseq <- length(seqx1x2)
MZ <- matrix(nrow = lseq, ncol = lseq)

for (i in 1:lseq) {
  for (j in 1:lseq) {
    x1 <- seqx1x2[i]
    x2 <- seqx1x2[j]
    x1x2 <- as.matrix(cbind(1, x1, x2))  # Adiciona o bias
    h1 <- cbind(1, tanh(x1x2 %*% Z))      # Calcula a ativação na camada oculta
    MZ[i, j] <- sign(h1 %*% W)            # Calcula a previsão
  }
}

main_title <- paste("Superfície de Separação (XOR) p =", p)

contour(seqx1x2, seqx1x2, MZ, nlevels = 1,
        xlim = range(seqx1x2), ylim = range(seqx1x2),
        xlab = "X1", ylab = "X2",
        main = main_title, drawlabels = FALSE, col = "black")

# Adicionar os pontos de validação ao gráfico
points(xin_validacao[yin_validacao == -1, 1], xin_validacao[yin_validacao == -1, 2], col = "red", pch = 19)
points(xin_validacao[yin_validacao == 1, 1], xin_validacao[yin_validacao == 1, 2], col = "blue", pch = 19)

# Plotar a superfície de separação em 3D
persp3d(seqx1x2, seqx1x2, MZ, col = "red", 
        xlim = range(seqx1x2), ylim = range(seqx1x2), 
        zlim = c(min(MZ), max(MZ)),  # Definindo os limites do eixo Z
        xlab = "X1", ylab = "X2", zlab = "Classe")
