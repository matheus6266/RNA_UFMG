YELM <- function(xin, Z, W, par) {
  # Obter o número de colunas (dimensão) da matriz de entrada xin
  n <- dim(xin)[2]
  
  # Verificar o valor do parâmetro 'par' para ajustar a matriz xin
  if (par == 1) {
    # Adiciona uma coluna de 1s à matriz de entrada xin (bias)
    xin <- cbind(1, xin)
  }
  
  # Calcula a função de ativação (tangente hiperbólica) na camada oculta
  H <- tanh(xin %*% Z)
  
  # Adiciona uma coluna de 1s à matriz H para incluir o bias
  Haug <- cbind(1, H)
  
  # Calcula as previsões (classificação) usando o sinal do produto interno
  Yhat <- sign(Haug %*% W)
  
  # Retorna as previsões
  return(Yhat)
}
