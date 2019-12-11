#' Converts a list of Seurat objects into a list of Single Cell Experiment objects
#' Extends the native Seurat function "as.SingleCellExperiment" by adding the capability to pass lists of Seurat objects
#' 
#' @return A correctly initialized SingleCellExperiment object
#'
#' @author Tyler Brassel, Nelson Johansen, Gerald Quon
#' @references Johansen, N., Quon, G. scAlign: a tool for alignment, integration, and rare cell identification from scRNA-seq data.
#'             Genome Biol 20, 166 (2019) doi: https://doi.org/10.1186/s13059-019-1766-4
#' @seealso <https://github.com/quon-titative-biology/scAlign>
#'
#' @param seurat.obj List of Seurat objects to be converted to Single Cell Experiment objects.
#' @param genes.use A character vector of gene names which exhibit variable expression across all datasets
#'
#' @seealso \code{\link{ExtractGenes}}
#' @seealso \code{\link{SelectColors}}
#' @seealso \code{\link{DecoderVariance}}
#' @seealso \code{\link{MeanDecoderVariance}}
#' @seealso \code{\link{GetCharMetadata}}
#'
#' @examples
#'
#' sce.objects = as.SingleCellExperimentList(seurat.obj)
#' 
#' @import SingleCellExperiment
#' @import Seurat
#' @import purrr
#' 
#' @export
as.SingleCellExperimentList = function(seurat.obj, 
                                       genes.use = NULL){
  if(all(vapply(seurat.obj, class, character(1)) == "Seurat") == TRUE){
    # Sanity check that all gene names have a correponding row name in all listed Seurat objects
    check = logical(length = length(seurat.obj))
    for(i in 1:length(seurat.obj)){
      if(is_empty(genes.use)){
        genes.index = 1:length(rownames(seurat.obj[[i]]))
        genes.use <- genes.index
      }else if(!is_empty(genes.use) && is.character(genes.use)){
        genes.index = which(rownames(seurat.obj[[i]]) %in% genes.use)
    }
    check[i] <- (length(genes.index) == length(genes.use))}
    # If all gene names correspond, return a correctly initialized list of SCE objects
    if(all(check)){
      sce.objects = lapply(seurat.obj, function(seurat.obj){
      SingleCellExperiment(assays = list(counts = (GetAssayData(object = seurat.obj, slot = "counts"))[genes.index,],
                                         logcounts = (GetAssayData(object = seurat.obj, slot = "data"))[genes.index,],
                                         scale.data = (GetAssayData(object = seurat.obj, slot = "scale.data"))[genes.index,]),
                                         colData = seurat.obj[[]])})
      return(sce.objects)
    }else{
      stop("One or more gene names used does not match a row name in the Seurat object.")
    }
  }else{
    stop("Please pass in 1 or more Seurat objects to be converted to a list of SingleCellExperiment objects.")
  }
}

#' Extracts common set of genes across all datasets
#'
#' @return A character vector of gene names which exhibit varible expression across all datasets
#'
#' @param seurat.obj1 Either a preprocessed Seurat object or a list of Seurat objects
#' @param seurat.obj2 A preprocessed Seurat object (optional, for simple binary conditions)
#' 
#' @seealso \code{\link{as.SingleCellExperimentList}}
#' @seealso \code{\link{SelectColors}}
#' @seealso \code{\link{DecoderVariance}}
#' @seealso \code{\link{MeanDecoderVariance}}
#' @seealso \code{\link{GetCharMetadata}}
#' 
#' @examples
#' for (i in 1:length(seurat.list)) {
#'   seurat.list[[i]] <- NormalizeData(seurat.list[[i]])
#'   seurat.list[[i]] <- ScaleData(seurat.list[[i]], do.scale=T, do.center=T, display.progress=T)
#'   seurat.list[[i]] <- FindVariableFeatures(seurat.list[[i]], nFeatures = 3000)
#' }
#' genes.use <- ExtractGenes(seurat.list)
#' 
#' @import SingleCellExperiment
#' @import Seurat
#' 
#' @export
ExtractGenes = function(seurat.obj1,
                        seurat.obj2 = NULL){
  if (all(vapply(seurat.obj1, class, character(1)) == "Seurat") == FALSE) {
    stop("Unsupported or inconsistent input type(s): Must be Seurat objects")
  }else if(is.list(seurat.obj1) && is.null(seurat.obj2)){
    genes.use <- Reduce(intersect, lapply(seurat.obj1, function(seurat.obj1) VariableFeatures(seurat.obj1)))
  }else if(!is.null(seurat.obj1) && !is.null(seurat.obj2)){
    genes.use <- Reduce(intersect, list(VariableFeatures(seurat.obj1),
                                        VariableFeatures(seurat.obj2),
                                        rownames(seurat.obj1),
                                        rownames(seurat.obj2)))
  }else{
    stop("Please pass either two Seurat objects or a list of Seurat objects")
  }
  return(genes.use)
}

#' Creates a list of unique color values used for plotting
#'
#' @return A named vector of unique hexedecimal color values, either generated from a preselected
#'         vector of 20 unique colors, or from a sequence of colors in hsv colorspace.
#'
#' @param seurat.obj A singular preprocessed Seurat object
#' @param gradient Setting to TRUE will use a sequence of hsv colors instead of 20 unique colors,
#'                 useful for comparisons of more than 20 cell types.
#' @param value The Seurat metadata slot to generate colors for. Defaults to "celltype".
#'
#' @import SingleCellExperiment
#' @import Seurat
#' 
#' @seealso \code{\link{as.SingleCellExperimentList}}
#' @seealso \code{\link{ExtractGenes}}
#' @seealso \code{\link{DecoderVariance}}
#' @seealso \code{\link{MeanDecoderVariance}}
#' @seealso \code{\link{GetCharMetadata}}
#'
#' @examples
#' DimPlot(object = seurat.obj,
#'         reduction = "tsne",
#'         cols = SelectColors(seurat.obj),
#'         group.by = "celltype",
#'         label = TRUE,
#'         repel = TRUE)
#' 
#' @export
SelectColors <- function(seurat.obj, gradient = FALSE, value = "celltype"){
  names <- GetCharMetadata(seurat.obj, value = value)
  if(gradient == FALSE && length(names) <= 20){
    celltype_colors_unique <- c('#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                                '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                                '#bcf60c', '#fabebe', '#008080', '#000000',
                                '#800000', '#aaffc3', '#808000', '#ffd8b1',
                                '#000075', '#808080', '#e6beff', '#9a6324')
    colors_use <- celltype_colors_unique[1:length(names)]
  }else{
    colors_use <- hsv(seq(0, 1 - 1/(length(names)), length.out = (length(names))))
  }
  names(colors_use) <- names
  return(colors_use)
}

#' Creates a 2d array of variance values calculated across the 3rd decoder dimension of a 3d array of decoder matrices.
#' Unless specified, automatically calculates highly variable genes and orders the combined variance matrix in decreasing order, for later graphical analysis analysis.
#' 
#' @return A named vector of unique hexedecimal color values, either generated from a preselected
#'         vector of 20 unique colors, or from a sequence of colors in hsv colorspace.
#'
#' @param seurat.obj A singular preprocessed Seurat object
#' @param gradient Setting to TRUE will use a sequence of hsv colors instead of 20 unique colors,
#'                 useful for comparisons of more than 20 cell types.
#' @param names Setting to TRUE will return the celltype data slot as a variable "names"
#'
#' @seealso \code{\link{as.SingleCellExperimentList}}
#' @seealso \code{\link{ExtractGenes}}
#' @seealso \code{\link{SelectColors}}
#' @seealso \code{\link{MeanDecoderVariance}}
#' @seealso \code{\link{GetCharMetadata}}
#'
#' @examples
#' DimPlot(object = seurat.obj,
#'         reduction = "tsne",
#'         cols = SelectColors(seurat.obj),
#'         group.by = "celltype",
#'         label = TRUE,
#'         repel = TRUE)
#'         
#' @import SingleCellExperiment
#' @import Seurat
#' 
#' @export
DecoderVariance <- function(all_data_combined,
                            seurat.obj,
                            genes.use,
                            output.length,
                            decreasing = TRUE,
                            return.genes = TRUE){
  all_data_var <- array(data = 0.0, dim = c(as.numeric(dim(all_data_combined)[1]),
                                            as.numeric(dim(all_data_combined)[2])))
  rownames(all_data_var) <- GetCharMetadata(seurat.obj = seurat.obj, value = "celltype", unique = FALSE)
  colnames(all_data_var) <- genes.use
  for(i in 1:dim(all_data_combined)[1]){
    for(j in 1:dim(all_data_combined)[2]){
      data <- c(all_data_combined[i, j, ])
      var_data <- var(data)
      all_data_var[i, j] <- var_data
    }
  }
  rm(var_data, data, i, j)
  if(return.genes == FALSE){
    list <- list(all_data_var)
    return(list)
  }else{
    high_var_genes <- sort(MeanDecoderVariance(all_data_var = all_data_var), decreasing = decreasing,index.return = T)$ix[1:output.length]
    list <- list(all_data_var, high_var_genes)
    return(list)
  }
}

#' Calculate mean decoder variance
#' @return A vector of mean data values for each column of the "all_data_var" parameter.
#'
#' @param all_data_var A matrix of variance values calculated across the decoder dimension.
#'
#' @seealso \code{\link{as.SingleCellExperimentList}}
#' @seealso \code{\link{ExtractGenes}}
#' @seealso \code{\link{SelectColors}}
#' @seealso \code{\link{DecoderVariance}}
#' @seealso \code{\link{GetCharMetadata}}
#' 
#' @export
MeanDecoderVariance <- function(all_data_var){
  all_mean_data <- vector()
  for(j in 1:dim(all_data_var)[2]){
    data <- all_data_var[, j]
    all_mean_data[j] <-  mean(data)
  }
  return(all_mean_data)
}

#' Fetch metadata as a character vector
#' @return A character vector of the unique values accessed by the Seurat function FetchData
#'
#' @param value The name of the internal metadata object the function should access
#' @param seurat.obj A seurat object
#' 
#' @seealso \code{\link{as.SingleCellExperimentList}}
#' @seealso \code{\link{ExtractGenes}}
#' @seealso \code{\link{SelectColors}}
#' @seealso \code{\link{DecoderVariance}}
#' @seealso \code{\link{MeanDecoderVariance}}
#'
#' @import Seurat
#' 
#' @export
GetCharMetadata <- function(seurat.obj, value = "celltype", unique = TRUE){
  if (unique == TRUE){
    cell_names <- unlist(c(unique(FetchData(object = seurat.obj, vars = c(value)))))
  }else if (unique == FALSE){
    cell_names <- unlist(c(FetchData(object = seurat.obj, vars = c(value))))
  }
return(cell_names)
}