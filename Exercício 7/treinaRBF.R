
treinaRBF <- function(xin, yin, p, r) {
  
  library(corpcor)
  
  # Função que calcula a variável radial
  radialvar <- function(x, m, invK) {
    exp(-0.5 * (t(x - m) %*% invK %*% (x - m)))
  }
  
  N <- dim(xin)[1]  # número de amostras
  n <- dim(xin)[2]  # dimensão de entrada (deve ser maior que 1)
  
  xin <- as.matrix(xin)  # garante que xin seja matriz
  yin <- as.matrix(yin)  # garante que yin seja matriz
  
  xclust <- kmeans(xin, p)  # faz a partição de xin em p regiões
  
  
  
  m <- as.matrix(xclust$centers)  # matriz com as médias das p regiões
  
  covic <- r * diag(n)  # matriz de covariâncias para todos os p centros
  inv_covi <- (1 / r) * diag(n)  # inversa da matriz de covariâncias
  
  H <- matrix(nrow = N, ncol = p)
  
  for (j in 1:N) {
    for (i in 1:p) {
      mi <- m[i, ]
      H[j, i] <- radialvar(xin[j, ], mi, inv_covi)  # projeção de xin
    }
  }
  
  Haug <- cbind(1, H)
  W <- pseudoinverse(Haug) %*% yin  # cálculo do vetor w
  
  return(list(m = m, covi = covic, r = r, W = W, H = H))

}
