#'  Object to store embeddings
#'
#' @return scAlignEmbedding object
#'
#' @keywords internal
scAlignEmbedding = setClass(
    ## Class name
    "scAlignEmbedding",
    ## Define the slots
    slots = c(embedding = "ANY",
              method.results = "list",
              type = "character")
)## End class def

#'  Initialize object of class
#'
#' @return Initialized scAlignEmbedding object
#'
#' @keywords internal
scAlignEmbeddingCreateObject = function(embedding, method.results, type){
  object = scAlignEmbedding(embedding=embedding, method.results=method.results, type=type)
  return(object)
}

#' scAlign S4 classs
#'
#' Holds input data and resulting data from scAlign.
#'
#' @return scAlign object
#'
#' @slot raw.data List containing unaligned (raw) datasets.
#' @slot scale.data List containing unaligned (raw) datasets.
#' @slot reduced.data List containing input data after an intial step of dimensionality reduction.
#' @slot aligned.data List containing aligned datasets.
#' @slot projected.data List containing the set of projections for each dataset.
#' @slot genes.use Vector of genes used to subset input data.
#' @slot meta.data data.frame containing dataset identifiers and cell information.
#' @slot options data.frame of parameters essential to neural network training.
#' @slot arguments data.frame recording arguments for scAlign function.
#' @slot project.name Name for the scAlign project.
#' @slot version Current version of scAlign.
#'
#' @export
scAlignClass = setClass(
    ## Class name
    "scAlignClass",
    ## Define the slots
    slots = c(raw.data = "list",
              scale.data = "list",
              reduced.data = "list",
              aligned.data = "list",
              projected.data = "list",
              genes.use = "character",
              meta.data = "data.frame",
              options = "data.frame",
              arguments = "data.frame",
              project.name = "character",
              version = "character")
)## End class def

#' Creates scAlign object
#'
#' @return Initialized scAlign object
#'
#' @param objects List of Seurat or Matrix objects; sample x feature.
#' @param labels List of labels for each object.
#' @param genes.use Genes to use during analysis.
#' @param meta.data Additional meta.data to add.
#' @param pca.reduce Initial step of dimensionality be performced by PCA.
#' @param pcs.compute Number of PCs to retrain for alignment.
#' @param cca.reduce Initial step of dimensionality be performced by CCA.
#' @param ccs.compute Number of CCs to retrain for alignment.
#' @param data.use Specificies which data to use from a Seurat object.
#' @param project.name Name for current scAlign project.
#'
#' @import irlba
#' @import Seurat
#' @import methods
#'
#' @examples
#'
#' library(Seurat)
#'
#' ## Input data, 1000 genes x 100 cells
#' data = matrix(sample.int(10000, 1000*100, TRUE), 1000, 100)
#' rownames(data) = paste0("gene", seq_len(1000))
#' colnames(data) = paste0("cell", seq_len(100))
#'
#' age    = c(rep("young",50), rep("old",50))
#' labels = c(c(rep("type1",25), rep("type2",25)), c(rep("type1",25), rep("type2",25)))
#'
#' ctrl.data = data[,which(age == "young")]
#' stim.data = data[,which(age == "old")]
#'
#' ctrl = CreateSeuratObject(raw.data = ctrl.data, project = "MOUSE_AGE", min.cells = 0)
#' ctrl@meta.data$stim  = "YOUNG"
#' ctrl@meta.data$label = labels[which(age == "young")]
#' ctrl = ScaleData(ctrl, do.scale=FALSE, do.center=FALSE, scale.max=50, display.progress = TRUE)
#'
#' stim = CreateSeuratObject(raw.data = stim.data, project = "MOUSE_AGE", min.cells = 0)
#' stim@meta.data$stim = "OLD"
#' stim@meta.data$label = labels[which(age == "old")]
#' stim = ScaleData(stim, do.scale=FALSE, do.center=FALSE, scale.max=50, display.progress = TRUE)
#'
#' ## Build the scAlign class object and compute PCs
#' scAlignHSC = scAlignCreateObject(objects = list("YOUNG"=ctrl, "OLD"=stim),
#'                                  labels = list(ctrl@meta.data$label, stim@meta.data$label),
#'                                  pca.reduce = FALSE,
#'                                  pcs.compute = 50,
#'                                  cca.reduce = FALSE,
#'                                  ccs.compute = 15,
#'                                  project.name = "scAlign_example")
#'
#' @export
scAlignCreateObject = function(objects,
                               labels = NULL,
                               genes.use = NULL,
                               meta.data = NULL,
                               pca.reduce = TRUE,
                               pcs.compute = 50,
                               cca.reduce = TRUE,
                               ccs.compute = 15,
                               data.use = "scale.data",
                               project.name = "scAlignProject"){

  ## Grab version of scAlign being run
  version = packageVersion("scAlign")

  ## Ensure data list has names
  if(is.null(names(objects))){ names(objects) = paste0("object", seq_len(length(objects))) }

  ## Input data handling
  raw.data = list(); group = c();
  if(all(vapply(objects, class, character(1)) == "seurat")){
    ## Load in seurat data from correct data type
    print("Using Seurat objects for scAlign.")
    for(name in names(objects)){ raw.data[[name]] = slot(objects[[name]], data.use);
                                 group = c(group, rep(name, ncol(slot(objects[[name]], data.use)))) }
  }else{ stop("Unknown or inconsistent input type(s)."); }

  ## Input data handling
  scale.data = list();
  if(all(vapply(objects, class, character(1)) == "seurat")){
    ## Load in seurat data from correct data type
    print("Using Seurat objects for scAlign.")
    for(name in names(objects)){ scale.data[[name]] = slot(objects[[name]], data.use); }
  }else{ stop("Unknown or inconsistent input type(s)."); }

  ## Reduce to provided gene set if supplied, otherwise use rownames
  if(!is.null(genes.use)){
    for(name in names(scale.data)){
      if(all(genes.use %in% rownames(scale.data[[name]]))){
        scale.data[[name]] = scale.data[[name]][genes.use,]
      }else{ stop(paste0("Provided genes.use contains elements not in rownames of ", name, ".")); }
    }
  }else{ genes.use = ''}

  ## Create meta.data object
  meta.data = data.frame(group = character(),
                         labels = character(),
                         stringsAsFactors=FALSE)
  meta.data[seq_len(length(group)),] = c(group, unlist(labels))

  ## Reduce to top 50 pcs
  emb = list()
  if(pca.reduce == TRUE){
    print(paste("Computing partial PCA for top ", pcs.compute, " PCs."))
    combined.data = t(Reduce(cbind, scale.data))
    pca_results = irlba(A = combined.data, nv = pcs.compute)
    pca_data = pca_results$u
    rownames(pca_data) = rownames(combined.data)
    colnames(pca_data) = paste0("PC", seq_len(pcs.compute))
    emb[["PCA"]] = scAlignEmbeddingCreateObject(pca_data, pca_results, "PCA")
  }

  ## Reduce to top ccs
  if(cca.reduce == TRUE){
    print(paste("Computing CCA using Seurat."))
    if(!all(vapply(objects, class, character(1)) == "seurat")){
      for(name in names(objects)){
        objects[[name]] = CreateSeuratObject(raw.data = objects[[name]], project = name, min.cells = 0)
        objects[[name]] = ScaleData(objects[[name]], do.center=FALSE, do.scale=FALSE)
      }
    }
    combined = RunCCA(objects[[names(objects)[1]]], objects[[names(objects)[2]]], num.cc=ccs.compute, scale.data=TRUE)
    emb[["CCA"]] = scAlignEmbeddingCreateObject(combined@dr$cca@cell.embeddings, list(combined@dr$cca@gene.loadings), "CCA")
  }

  arguments = data.frame(encoder.data=character(),
                         decoder.data=character(),
                         supervised=character(),
                         run.encoder=logical(),
                         run.decoder=logical(),
                         log.dir=character(),
                         log.results=logical(),
                         device=character(), stringsAsFactors=FALSE)

  options = data.frame(steps=integer(),
                       batch.size=integer(),
                       learning.rate=numeric(),
                       log.every=integer(),
                       architecture=character(),
                       num.dim=integer(),
                       perplexity=integer(),
                       norm=logical(),
                       early.stop=logical(),
                       seed=integer(), stringsAsFactors=FALSE)

  scAlignObject = scAlignClass(raw.data = raw.data,
                               scale.data = scale.data,
                               reduced.data = emb,
                               genes.use = genes.use,
                               options = options,
                               arguments = arguments,
                               meta.data = meta.data,
                               project.name = project.name,
                               version = as.character(version))
  return(scAlignObject)
}

#' Set training options
#'
#' Defines parameters for optimizer and training procedure.
#'
#' @return Options data.frame
#'
#' @param steps (default: 15000) Number of training iterations for neural networks.
#' @param batch.size (default: 150) Number of input samples per training batch.
#' @param learning.rate (default: 1e-4) Initial learning rate for ADAM.
#' @param log.every (default: 5000) Number of steps before saving results.
#' @param architecture (default: "small") Network function name for scAlign.
#' @param num.dim (default: 32) Number of dimensions for joint embedding space.
#' @param perplexity (default: 30) Determines the neighborhood size for each sample.
#' @param norm (default: TRUE) Normalize the data matrix prior to scAlign.
#' @param early.stop (default: TRUE) Early stopping during network training.
#' @param seed (default: 1245) Sets graph level random seed in tensorflow.
#
#' @examples
#'
#' options=scAlignOptions(steps=15000,
#'                        log.every=5000,
#'                        early.stop=FALSE,
#'                        architecture="large")
#'
#' @export
scAlignOptions = function(steps = 15000, batch.size = 150,
                          learning.rate = 1e-4, log.every = 5000,
                          architecture="small",
                          num.dim = 32, perplexity = 30,
                          norm = TRUE, early.stop = TRUE,
                          seed = 1234){

     valid_opts = c("steps", "batch.size", "learning.rate", "log.every", "architecture",
                   "num.dim", "perplexity", "norm", "early.stop", "seed")
     opts = data.frame(steps = steps,
                       batch.size = batch.size,
                       learning.rate = learning.rate,
                       log.every = log.every,
                       architecture = architecture,
                       num.dim = num.dim,
                       perplexity = perplexity,
                       norm = norm,
                       early.stop = early.stop,
                       seed = seed, stringsAsFactors=FALSE)
    colnames(opts) = valid_opts

    if(!is.null(options)){
      if(all(names(options) %in% valid_opts)){
        ## Populate options with user supplied parameters
        for(name in names(options)){
          opts[,name] = options[[name]]
        }
      }else{ stop(paste0("These provided options are not valid: ", names(options)[which(!names(options) %in% valid_opts)])) }
    }
    return(opts)
}

#' Record aguments passed to scAlign
#'
#' @return Arguments data.frame
#'
#' @keywords internal
scAlignArguments = function(object,
                            encoder.data,
                            decoder.data,
                            supervised,
                            run.encoder,
                            run.decoder,
                            log.dir,
                            log.results,
                            device){

    args = data.frame(encoder.data=encoder.data,
                      decoder.data=decoder.data,
                      supervised=supervised,
                      run.encoder=run.encoder,
                      run.decoder=run.decoder,
                      log.dir=normalizePath(log.dir, mustWork=FALSE),
                      log.results=log.results,
                      device=device, stringsAsFactors=FALSE)
    colnames(args) = c("encoder.data", "decoder.data", "supervised", "run.encoder", "run.decoder", "log.dir", "log.results", "device")

    object = setArgs(object, args)
    return(object)
}

#' Set a new embedding
#'
#' @return scAlign object with embedding slot set
#'
#' Adds additional dimensionality reduced data to scAlign class
#'
#' @param object scAlign S4 class object
#' @param emb Matrix of cells x features
#' @param emb.name Name of dimensionality reduction
#' @param emb.results Other results from dimensionality reduction method.
#'
#' @keywords internal
scAlignSetEmbedding = function(object, emb, emb.name, emb.results=NULL){
  object = setEmbedding(object, emb, emb.results, emb.name)
  return(object)
}

################################################################################
## SETTER FUNCTIONS FOR scAlignClass
################################################################################

#' Generic assigns aligned data to S4 slot
#'
#' @return scAlign class with alignment slot set
#'
#' @keywords internal
setGeneric(name="setAligned",
           def=function(self, aligned.data, data.use){ standardGeneric("setAligned") })

#' Method assigns aligned data to S4 slot
#'
#' @return scAlign class with alignment slot set
#'
#' @keywords internal
setMethod(f="setAligned",
          signature="scAlignClass",
          definition=function(self, aligned.data, data.use){
                          rownames(aligned.data) = paste0("sample", seq_len(nrow(aligned.data)))
                          colnames(aligned.data) = paste0("dim", seq_len(ncol(aligned.data)))
                          self@aligned.data[[data.use]] = scAlignEmbeddingCreateObject(aligned.data, list(), data.use)
                          return(self)})

#' Generic assigns projected data to S4 slots
#'
#' @return scAlign class with projection slot set
#'
#' @keywords internal
setGeneric(name="setProjected",
           def=function(self, projected.data, data.name, projection.name){ standardGeneric("setProjected") })

#' Method assigns projected data to S4 slots
#'
#' @return scAlign class with projection slot set
#'
#' @keywords internal
setMethod(f="setProjected",
          signature="scAlignClass",
          definition=function(self, projected.data, data.name, projection.name){
                          rownames(projected.data) = paste0("sample", seq_len(nrow(projected.data)))
                          colnames(projected.data) = rownames(self@scale.data[[1]])
                          if(is.null(self@projected.data[[data.name]])){
                            self@projected.data[[data.name]] = list()
                          }
                          self@projected.data[[data.name]][[projection.name]] = scAlignEmbeddingCreateObject(projected.data, list(), projection.name)
                          return(self)})

#' Generic assigns projected data to S4 slots
#'
#' @return scAlign class with options slot set
#'
#' @keywords internal
setGeneric(name="setOptions",
           def=function(self, options){ standardGeneric("setOptions") })

#' Method assigns projected data to S4 slots
#'
#' @return scAlign class with options slot set
#'
#' @keywords internal
setMethod(f="setOptions",
          signature="scAlignClass",
          definition=function(self, options){
                          self@options = options
                          return(self)})

#' Generic assigns projected data to S4 slots
#'
#' @return scAlign class with arguments slot set
#'
#' @keywords internal
setGeneric(name="setArgs",
           def=function(self, args){ standardGeneric("setArgs") })

#' Method assigns projected data to S4 slots
#'
#' @return scAlign class with arguments slot set
#'
#' @keywords internal
setMethod(f="setArgs",
          signature="scAlignClass",
          definition=function(self, args){
                          self@arguments = args
                          return(self)})

#' Generic assigns aligned data to S4 slot
#'
#' @return scAlign class with reduced.data slot set
#'
#' @keywords internal
setGeneric(name="setEmbedding",
           def=function(self, emb, emb.results, emb.name){ standardGeneric("setEmbedding") })

#' Method assigns aligned data to S4 slot
#'
#' @return scAlign class with reduced.data slot set
#'
#' @keywords internal
setMethod(f="setEmbedding",
          signature="scAlignClass",
          definition=function(self, emb, emb.results, emb.name){
                          if(is.null(emb.results)){
                            emb.results = list();
                          }else if(!is.list(emb.results)){
                            emb.results = list(emb.results)
                          }
                          self@reduced.data[[emb.name]] = scAlignEmbeddingCreateObject(emb, emb.results, emb.name)
                          return(self)})

#' Method prints S4 class nicely
#'
#' @return Modified print statment for scAlign class objects
#'
#' @keywords internal
setMethod(f="show",
          signature="scAlignClass",
          definition = function(object){
              cat(paste0("An object of class scAlign(", object@version ,") in project ", object@project.name, "\n"))
              cat(paste0("  ", length(object@scale.data), " datasets containing ", nrow(object@meta.data), " samples.\n"))
          })

# #' Generic assigns metadata to S4 slot
# #'
# #' @keywords internal
# setGeneric(name="setMetaData",
#            def=function(self, projected_obj1, projected_obj2){ standardGeneric("setMetaData") })
#
# #' Method assigns metadata to S4 slot
# #'
# #' @keywords internal
# # setMethod(f="setMetaData",
# #           signature="scAlignClass",
# #           definition=function(self, projected_obj1, projected_obj2){
# #                           self@projected_obj1 = projected_obj1
# #                           self@projected_obj2 = projected_obj2
# #                           return(self)})
