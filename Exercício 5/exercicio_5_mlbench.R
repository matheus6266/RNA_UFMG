rm(list = ls())
library(corpcor)
library(rgl)
library(mlbench)
library(rgl)

source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 5\\trainELM.R")
source("C:\\Users\\mathe\\OneDrive\\Pós Graduação UFMG\\Exercício 5\\YELM.R")

####################################### mlbench.2dnormals(200)
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
     main = "Estrutura dos Dados de Treino", xlab = "Feature 1", ylab = "Feature 2")
points(xin_treino[yin_treino == 1, 1], xin_treino[yin_treino == 1, 2], col = "blue", pch = 19)

# Plotar os dados de validação com as cores ajustadas
plot(xin_validacao[yin_validacao == -1, 1], xin_validacao[yin_validacao == -1, 2], col = "red", pch = 19,
     xlim = range(xin_validacao[, 1]), ylim = range(xin_validacao[, 2]),
     main = "Estrutura dos Dados de Validação", xlab = "Feature 1", ylab = "Feature 2")
points(xin_validacao[yin_validacao == 1, 1], xin_validacao[yin_validacao == 1, 2], col = "blue", pch = 19)

# Treinar a ELM usando a função treinaELM
p <- 5  # Número de neurônios na camada oculta, você pode ajustar conforme necessário
par <- 1  # Definindo o parâmetro 'par' (1 para adicionar bias)

#xin_treino <- scale(xin_treino)  # Normaliza os dados de treino
#xin_validacao <- scale(xin_validacao)  # Normaliza os dados de validação

modelo_elm <- treinaELM(xin_treino, yin_treino, p, par)

# Acessar os pesos da camada de saída (W)
W <- modelo_elm[[1]]

# Acessar a matriz de ativações da camada oculta (H)
H <- modelo_elm[[2]]

# Acessar a matriz de pesos aleatórios (Z)
Z <- modelo_elm[[3]]

# Agora, W, H e Z são objetos separados que você pode usar

# Fazer as previsões novamente
predicoes <- YELM(xin_validacao, Z, W, par)


# Ver as primeiras previsões
print(head(predicoes))

# Calcular a acurácia do modelo
acuracia <- mean(predicoes == yin_validacao)
print(paste("Acurácia: ", round(acuracia * 100, 2), "%"))

# Definir a grade para calcular a superfície de separação
seqx1x2 <- seq(-5, 5, 0.1)
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

main_title <- paste("Superfície de Separação (XOR) p =", p)

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
