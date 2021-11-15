#' Seurat V2 MultiCCA
#'
#' @return CCA results
#'
#' @param mat.list List of matrix objects
#' @param niter Optimization parameter
#' @param num.ccs Number of dimensions
#' @param standardize To standardize data
#'
#' @import Seurat
#' @import irlba
#'
#' @keywords internal
RunMultiCCA <- function(
  mat.list,
  niter = 25,
  num.ccs = 10,
  standardize = TRUE
) {
  num.sets <- length(mat.list)
  if(standardize){
    for (i in 1:num.sets){
      mat.list[[i]] <- scale(mat.list[[i]], TRUE, TRUE)
    }
  }
  ws <- list()
  for (i in 1:num.sets){
    ws[[i]] <- irlba(mat.list[[i]], nv = num.ccs)$v[, 1:num.ccs, drop = FALSE]
  }
  ws.init <- ws
  ws.final <- list()
  cors <- NULL
  for(i in 1:length(ws)){
    ws.final[[i]] <- matrix(0, nrow=ncol(mat.list[[i]]), ncol=num.ccs)
  }
  for (cc in 1:num.ccs){
    print(paste0("Computing CC ", cc))
    ws <- list()
    for (i in 1:length(ws.init)){
      ws[[i]] <- ws.init[[i]][, cc]
    }
    cur.iter <- 1
    crit.old <- -10
    crit <- -20
    storecrits <- NULL
    while(cur.iter <= niter && abs(crit.old - crit)/abs(crit.old) > 0.001 && crit.old !=0){
      crit.old <- crit
      crit <- GetCrit(mat.list, ws, num.sets)
      storecrits <- c(storecrits, crit)
      cur.iter <- cur.iter + 1
      for(i in 1:num.sets){
        ws[[i]] <- UpdateW(mat.list, i, num.sets, ws, ws.final)
      }
    }
    for(i in 1:length(ws)){
      ws.final[[i]][, cc] <- ws[[i]]
    }
  }
  cca.data <- ws.final[[1]]
  for(i in 2:length(mat.list)){
    cca.data <- rbind(cca.data, ws.final[[i]])
  }
  cca.data <- apply(cca.data, MARGIN = 2, function(x){
    if(sign(x[1]) == -1) {
      x <- x * -1
    }
    return(x)
  })
  return(cca.data)
}

#' Seurat V2 MultiCCA
#'
#' @return CCA results
#'
#' @param mat.list List of matrix objects
#' @param i Optimization parameter
#' @param num.sets Number of sets
#' @param ws optim
#' @param ws.final optim
#'
#' @keywords internal
UpdateW <- function(mat.list, i, num.sets, ws, ws.final){
  tots <- 0
  for(j in (1:num.sets)[-i]){
    diagmat <- (t(ws.final[[i]])%*%t(mat.list[[i]]))%*%(mat.list[[j]]%*%ws.final[[j]])
    diagmat[row(diagmat)!=col(diagmat)] <- 0
    tots <- tots + t(mat.list[[i]])%*%(mat.list[[j]]%*%ws[[j]]) - ws.final[[i]]%*%(diagmat%*%(t(ws.final[[j]])%*%ws[[j]]))
  }
  w <- tots/l2n(tots)
  return(w)
}

#' Seurat V2 MultiCCA
#'
#' @return crit values
#'
#' @param mat.list List of matrix objects
#' @param ws optim
#' @param num.sets Number of sets
#'
#' @keywords internal
GetCrit <- function(mat.list, ws, num.sets){
  crit <- 0
  for(i in 2:num.sets){
    for(j in 1:(i-1)){
      crit <- crit + t(ws[[i]])%*%t(mat.list[[i]])%*%mat.list[[j]]%*%ws[[j]]
    }
  }
  return(crit)
}

#' Seurat V2 MultiCCA
#'
#' @return l2n
#'
#' @param vec vector
#'
#' @keywords internal
l2n <- function(vec){
  a <- sqrt(sum(vec^2))
  if(a==0){
    a <- .05
  }
  return(a)
}
