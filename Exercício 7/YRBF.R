YRBF <- function(xin, modRBF) {
  
  # Função que calcula a variável radial
  radialvar <- function(x, m, invK) {
    exp(-0.5 * (t(x - m) %*% invK %*% (x - m)))
  }
  
  N <- dim(xin)[1]  # número de amostras
  n <- dim(xin)[2]  # dimensão de entrada (deve ser maior que 1)
  
  m <- as.matrix(modRBF[[1]])  # matriz de médias
  covi <- modRBF[[2]]  # matriz de covariâncias
  inv_covi <- (1 / modRBF[[3]]) * diag(n)  # inversa da matriz covi
  
  Htr <- modRBF[[5]]  # projeção para o conjunto de treinamento
  p <- ncol(Htr)  # número de funções radiais
  
  W <- modRBF[[4]]  # vetor de pesos
  
  xin <- as.matrix(xin)  # garante que xin seja matriz
  
  H <- matrix(nrow = N, ncol = p)
  for (j in 1:N) {
    for (i in 1:p) {
      mi <- m[i, ]
      H[j, i] <- radialvar(xin[j, ], mi, inv_covi)  # projeção de xin
    }
  }
  
  Haug <- cbind(1, H)
  Yhat <- Haug %*% W  # saída
  
  return(Yhat)
}
