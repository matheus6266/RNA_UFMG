trainperceptron <- function(xin, yd, eta, tol, maxepocas, par){
  
  dimxin <- dim(xin)       # Dimensão do número de dados
  N <- dimxin[1]           # Número de amostras
  n <- dimxin[2]           # Dimensão de entrada
  
  if (par == 1){
    wt <- as.matrix(runif(n+1) - 0.5)
    xin <- cbind(-1, xin)
  }
  
  else wt <- as.matrix(runif(n) - 0.5)
  
  nepocas <- 0 # Contador de Épocas
  eepoca <- tol + 1 #Acumuladore de erro de épocas
  evec <- matrix(nrow = 1, ncol = maxepocas) # Vetor de erros
  
  while ((nepocas < maxepocas) && (eepoca > tol)) {
    
    ei2 <- 0
    
    # Sequência Aleatória de treinamento
    
    xseq <- sample(N)
    
    for (i in 1:N) {
      
      irand <- xseq[i] #Amostra dado da sequência aleatória
      yhati <- 1.0*((xin[irand, ] %*% wt) >= 0) #Calcula a saída do perceptron
      ei <- yd[irand] - yhati
      dw <- eta*ei*xin[irand, ]
      wt <- wt + dw # Ajusta o vetor de pesos
      ei2 <- ei2 + (ei*ei) # Acumula erro por época
    }
    
    nepocas <- nepocas + 1 # Incrementa o número de épocas
    evec[nepocas] <- ei2/N
    eepoca <- evec[nepocas] # Armazena erro por época
  }
  
  # Retorna valores de pesos e de erros
  
  retlist <- list(wt, evec[1:nepocas])
  return(retlist)
  
}