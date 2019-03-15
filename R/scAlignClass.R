#' Creates scAlign object
#'
#' @return Initialized scAlign object
#'
#' @param sce.objects List of Seurat or Matrix objects; sample x feature.
#' @param labels List of labels for each object.
#' @param data.use Specificies which data to use from a Seurat object for dimensionality reduction.
#' @param pca.reduce Initial step of dimensionality be performced by PCA.
#' @param pcs.compute Number of PCs to retrain for alignment.
#' @param cca.reduce Initial step of dimensionality be performced by CCA.
#' @param ccs.compute Number of CCs to retrain for alignment.
#' @param meta.data Additional meta.data to add.
#' @param project.name Name for current scAlign project.
#'
#' @import SingleCellExperiment
#' @import irlba
#' @import Seurat
#' @import methods
#' @import utils
#'
#' @examples
#'
#' library(Seurat)
#' library(SingleCellExperiment)
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
#' ctrl = ScaleData(ctrl, do.scale=TRUE, do.center=TRUE, scale.max=50, display.progress = TRUE)
#'
#' stim = CreateSeuratObject(raw.data = stim.data, project = "MOUSE_AGE", min.cells = 0)
#' stim = ScaleData(stim, do.scale=TRUE, do.center=TRUE, scale.max=50, display.progress = TRUE)
#'
#' ## Build the SCE object for input to scAlign using Seurat preprocessing and variable gene selection
#' ctrl.sce <- SingleCellExperiment(
#'               assays = list(
#'                 counts = t(FetchData(ctrl, vars.all=rownames(data), use.raw=TRUE)),
#'                 scale.data = t(FetchData(ctrl, vars.all=rownames(data), use.scaled=TRUE)))
#'
#' stim.sce <- SingleCellExperiment(
#'               assays = list(
#'                 counts = t(FetchData(stim, vars.all=rownames(data), use.raw=TRUE)),
#'                 scale.data = t(FetchData(stim, vars.all=rownames(data), use.scaled=TRUE)))
#'
#' ## Build the scAlign class object and compute PCs
#' scAlignHSC = scAlignCreateObject(sce.objects = list("YOUNG"=ctrl.sce,
#'                                                     "OLD"=stim.sce),
#'                                  labels = list(cell_type[which(cell_age == "young")],
#'                                                cell_type[which(cell_age == "old")]),
#'                                  pca.reduce = TRUE,
#'                                  pcs.compute = 50,
#'                                  cca.reduce = TRUE,
#'                                  ccs.compute = 15,
#'                                  project.name = "scAlign_Kowalcyzk_HSC")
#'
#' @export
scAlignCreateObject = function(sce.objects,
                               labels = list(),
                               meta.data = NULL,
                               pca.reduce = TRUE,
                               pcs.compute = 20,
                               cca.reduce = TRUE,
                               ccs.compute = 15,
                               data.use = "scale.data",
                               project.name = "scAlignProject"){

  ## Grab version of scAlign being run
  version = packageVersion("scAlign")

  ## Ensure all objects are SCE
  if(all(vapply(sce.objects, class, character(1)) == "SingleCellExperiment") == FALSE){
    stop("Unsupported or inconsistent input type(s): Must be SingleCellExperiment objects")
  }

  ## Ensure data list has names
  if(is.null(names(sce.objects))){ names(sce.objects) = paste0("dataset", seq_len(length(sce.objects))) }
  if(!is.list(labels)) { stop("labels must be a list.") }

  if(length(sce.objects)==2){
    ## Combine data into a common SCE
    combined.sce = cbind(sce.objects[[names(sce.objects)[1]]], sce.objects[[names(sce.objects)[2]]])
    ## Determines dataset split
    colData(combined.sce)[,"group.by"] = c(rep(names(sce.objects)[1], ncol(sce.objects[[names(sce.objects)[1]]])),
                                                    rep(names(sce.objects)[2], ncol(sce.objects[[names(sce.objects)[2]]])))
    ## Set labels to be used in supervised training or defaults
    colData(combined.sce)[,"scAlign.labels"] = if(length(unlist(labels)) == 0) rep("NA", nrow(colData(combined.sce))) else unlist(labels)
    if(length(colData(combined.sce)[,"scAlign.labels"]) < nrow(colData(combined.sce))) { stop("Not enough labels for all cells.") }
    metadata(combined.sce)[["arguments"]] = data.frame(encoder.data=character(),
                                             decoder.data=character(),
                                             supervised=character(),
                                             run.encoder=logical(),
                                             run.decoder=logical(),
                                             log.dir=character(),
                                             log.results=logical(),
                                             device=character(), stringsAsFactors=FALSE)
    metadata(combined.sce)[["options"]]  = data.frame(steps=integer(),
                                            batch.size=integer(),
                                            learning.rate=numeric(),
                                            log.every=integer(),
                                            architecture=character(),
                                            num.dim=integer(),
                                            perplexity=integer(),
                                            norm=logical(),
                                            early.stop=logical(),
                                            seed=integer(), stringsAsFactors=FALSE)
  }else{
    stop("Implementation of multi (>2) dataset alignment still in progress.")
  }

  ## Ensure data to be used for dimensionality reduction is available
  if(is.element(data.use, names(assays(combined.sce))) == FALSE){ stop("data.use is not reachable in assays.") }

  ## Reduce to top 50 pcs
  if(pca.reduce == TRUE){
    print(paste("Computing partial PCA for top ", pcs.compute, " PCs."))
    pca_results = irlba(A = t(assay(combined.sce, data.use)), nv = pcs.compute)
    pca_mat = pca_results$u
    colnames(pca_mat) <- paste0("PC", seq_len(pcs.compute))
    reducedDim(combined.sce, "PCA") = pca_results$u
    metadata(combined.sce)[["PCA.output"]] = pca_results
  }

  ## Reduce to top ccs
  if(cca.reduce == TRUE){
    print(paste("Computing CCA using Seurat."))
    ## Run Seurat normalization
    for(name in names(sce.objects)){
      sce.objects[[name]] = CreateSeuratObject(raw.data = assay(sce.objects[[name]], data.use), project = name, min.cells = 0)
      sce.objects[[name]] = ScaleData(sce.objects[[name]], do.center=TRUE, do.scale=TRUE)
    }
    ## Reduce from gene to cc space
    combined = RunCCA(sce.objects[[names(sce.objects)[1]]],
                      sce.objects[[names(sce.objects)[2]]],
                      genes.use=rownames(combined.sce),
                      num.cc=ccs.compute,
                      scale.data=TRUE)
    ## Load dim reduction into SCE object
    reducedDim(combined.sce, "CCA") = GetCellEmbeddings(combined, "cca")
    metadata(combined.sce)[["CCA.output"]] = GetGeneLoadings(combined, reduction.type = "cca")
  }
  return(combined.sce)
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
#' @param norm (default: TRUE) Normalize the data mini batches while training scAlign (repeated).
#' @param full_norm (default: FALSE) Normalize the data matrix prior to scAlign (done once).
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
                          norm = TRUE, full_norm = FALSE, early.stop = TRUE,
                          seed = 1234){

     valid_opts = c("steps", "batch.size", "learning.rate", "log.every", "architecture",
                   "num.dim", "perplexity", "norm", "full_norm", "early.stop", "seed")
     opts = data.frame(steps = steps,
                       batch.size = batch.size,
                       learning.rate = learning.rate,
                       log.every = log.every,
                       architecture = architecture,
                       num.dim = num.dim,
                       perplexity = perplexity,
                       norm = norm,
                       full_norm = full_norm,
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
scAlignArguments = function(sce.object,
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
    return(args)
}
