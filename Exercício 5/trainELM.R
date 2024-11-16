treinaELM <- function(xin, yin, p, par) {
  # Carregar o pacote 'corpcor', necessário para a função de pseudoinversa
  library(corpcor)
  
  # Obter o número de colunas (dimensão) da matriz de entrada xin
  n <- dim(xin)[2]
  
  # Verificar o valor do parâmetro 'par' para ajustar a matriz xin e definir Z
  if (par == 1) {
    # Adiciona uma coluna de 1s à matriz de entrada xin (bias)
    xin <- cbind(1, xin)
    
    # Gera uma matriz Z com valores aleatórios uniformemente distribuídos
    # (n+1) linhas e p colunas, com valores no intervalo [-0.5, 0.5]
    Z <- matrix(runif((n + 1) * p, -0.5, 0.5), nrow = (n + 1), ncol = p)
  } else {
    # Gera a matriz Z sem adicionar a coluna de bias (n linhas e p colunas)
    Z <- matrix(runif(n * p, -0.5, 0.5), nrow = n, ncol = p)
  }
  
  # Calcula a função de ativação (tangente hiperbólica) na camada oculta
  H <- tanh(xin %*% Z)
  
  # Adiciona uma coluna de 1s à matriz H para incluir o bias
  Haug <- cbind(1, H)
  
  # Calcula os pesos da camada de saída usando a pseudoinversa
  W <- pseudoinverse(Haug) %*% yin
  
  # Retorna os pesos da camada de saída, a matriz H e a matriz Z
  return(list(W, H, Z))
}
