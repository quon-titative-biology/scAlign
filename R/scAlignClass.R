#' Creates scAlign object
#'
#' @return Initialized scAlign object
#'
#' @param sce.objects List of Seurat or Matrix objects; sample x feature.
#' @param genes.use Genes to use during PCA/CCA, all genes used as default.
#' @param labels List of labels for each object.
#' @param data.use Specificies which data to use from a Seurat object for dimensionality reduction.
#' @param pca.reduce Initial step of dimensionality be performced by PCA.
#' @param pcs.compute Number of PCs to retrain for alignment.
#' @param cca.reduce Initial step of dimensionality be performced by CCA.
#' @param ccs.compute Number of CCs to retrain for alignment.
#' @param cca.standardize Standardize the data matrices before CCA.
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
#' ctrlSCE <- SingleCellExperiment(
#'               assays = list(scale.data = data[,which(age == "young")]))
#'
#' stimSCE <- SingleCellExperiment(
#'               assays = list(scale.data = data[,which(age == "old")]))
#'
#' ## Build the scAlign class object and compute PCs
#' scAlignHSC = scAlignCreateObject(sce.objects = list("YOUNG"=ctrlSCE,
#'                                                     "OLD"=stimSCE),
#'                                  labels = list(labels[which(age == "young")],
#'                                                labels[which(age == "old")]),
#'                                  pca.reduce = TRUE,
#'                                  pcs.compute = 50,
#'                                  cca.reduce = TRUE,
#'                                  ccs.compute = 15,
#'                                  project.name = "scAlign_Kowalcyzk_HSC")
#'
#' @export
scAlignCreateObject = function(sce.objects,
                               genes.use = NULL,
                               labels = list(),
                               pca.reduce = FALSE,
                               pcs.compute = 20,
                               cca.reduce = FALSE,
                               ccs.compute = 15,
                               cca.standardize = TRUE,
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

  ## Combine data into a common SCE
  combined.sce = Reduce(cbind, sce.objects)

  ## If no gene set provided then use all genes in common across datasets
  if(is.null(genes.use)){
    genes.use = rownames(combined.sce)
  }

  ## Ensure all objects are SCE
  if(!is(combined.sce, "SingleCellExperiment")){
    stop("Unsupported or inconsistent input type(s): Must be SingleCellExperiment objects")
  }
  ## Determines dataset split
  group.by=c()
  for(name in names(sce.objects)){
    group.by = c(group.by, rep(name, ncol(sce.objects[[name]])))
  }
  colData(combined.sce)[,"group.by"] = group.by
  if(length(labels) == length(sce.objects)){
    ## Set cell labels
    tryCatch({
      ## Set labels to be used in supervised training or defaults
      colData(combined.sce)[,"scAlign.labels.orig"] = unlist(labels)
      colData(combined.sce)[,"scAlign.labels"] = if(length(unlist(labels)) == 0) rep("NA", nrow(colData(combined.sce))) else (as.integer(factor(unlist(labels)))-1)
      colData(combined.sce)[,"scAlign.labels"][is.na(colData(combined.sce)[,"scAlign.labels"])] = -1
    }, error=function(e){print("Error converting labels to factors."); stop(e);})
  }else{
      colData(combined.sce)[,"scAlign.labels"] = rep(0, ncol(combined.sce))
      print("Ignoring labels list which is either empty or less than the number of sce objects")
  }
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

  ## Ensure data to be used for dimensionality reduction is available
  if(is.element(data.use, names(assays(combined.sce))) == FALSE){ stop("data.use is not reachable in assays.") }

  ## Reduce to top 50 pcs
  if(pca.reduce == TRUE){
    print(paste("Computing partial PCA for top ", pcs.compute, " PCs."))
    pca_results = irlba(A = t(assay(combined.sce, data.use)[genes.use,]), nv = pcs.compute)
    pca_mat = pca_results$u
    colnames(pca_mat) <- paste0("PC", seq_len(pcs.compute))
    reducedDim(combined.sce, "PCA") = pca_results$u
    metadata(combined.sce)[["PCA.output"]] = pca_results
  }

  ## Reduce to top ccs
  if(cca.reduce == TRUE & (length(sce.objects) == 2)){
    print(paste("Computing CCA using Seurat."))
    seurat.objects = list()
    for(name in names(sce.objects)){
      seurat.objects[[name]] = CreateSeuratObject(assay(sce.objects[[name]], data.use), project = name)
      seurat.objects[[name]] = ScaleData(seurat.objects[[name]], do.center=TRUE, do.scale=TRUE)
    }
    ## Functional changes due to seurat version
    if(packageVersion("Seurat") >= 3.0){
      ## Reduce from gene to cc space
      combined = RunCCA(seurat.objects[[names(sce.objects)[1]]],
                        seurat.objects[[names(sce.objects)[2]]],
                        features=genes.use,
                        num.cc=ccs.compute)
      ## Load dim reduction into SCE object
      reducedDim(combined.sce, "CCA") =  Embeddings(combined, reduction = "cca")
      metadata(combined.sce)[["CCA.output"]] = Loadings(combined, reduction = "cca")
    }else{
      ## Reduce from gene to cc space
      combined = RunCCA(seurat.objects[[names(sce.objects)[1]]],
                        seurat.objects[[names(sce.objects)[2]]],
                        genes.use=genes.use,
                        num.cc=ccs.compute)
      ## Load dim reduction into SCE object
      reducedDim(combined.sce, "CCA") = GetCellEmbeddings(combined, "cca")
      metadata(combined.sce)[["CCA.output"]] = GetGeneLoadings(combined, reduction.type = "cca")
    }
  }else if(cca.reduce == TRUE & length(sce.objects) > 2){
    ## Run Multi CCA for input to scAlign
    multi.cca.input <- lapply(sce.objects, function(seurat.obj){
        assay(seurat.obj, slot='scale.data')[genes.use,]
    })
    reducedDim(combined.sce, "MultiCCA") = RunMultiCCA(multi.cca.input, num.ccs = ccs.compute, standardize = cca.standardize)
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
#' @param steps.decoder Number of training iterations for neural networks.
#' @param batch.size (default: 150) Number of input samples per training batch.
#' @param learning.rate (default: 1e-4) Initial learning rate for ADAM.
#' @param log.every (default: 5000) Number of steps before saving results.
#' @param architecture (default: "small") Network function name for scAlign.
#' @param batch.norm.layer (default: FALSE) Include batch normalization in the network structure.
#' @param dropout.layer (default: TRUE) Include dropout in the network.
#' @param num.dim (default: 32) Number of dimensions for joint embedding space.
#' @param perplexity (default: 30) Determines the neighborhood size for each sample.
#' @param betas (default: 0) Sets the bandwidth of the gaussians to be the same if > 0. Otherwise per cell beta is computed.
#' @param norm (default: TRUE) Normalize the data mini batches while training scAlign (repeated).
#' @param full.norm (default: FALSE) Normalize the data matrix prior to scAlign (done once).
#' @param early.stop (default: TRUE) Early stopping during network training.
#' @param walker.loss (default: TRUE) Add walker loss to model.
#' @param reconc.loss (default: FALSE) Add reconstruction loss to model during alignment.
#' @param walker.weight (default: 1.0) Weight on walker loss component
#' @param classifier.weight (default: 1.0) Weight on classifier loss component
#' @param classifier.delay (default: NULL) Delay classifier component of loss function until specific training step. Defaults to (2/3)*steps.
#' @param gpu.device (default: '0') Which gpu to use.
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
scAlignOptions = function(steps = 15000, steps.decoder=10000, batch.size = 150,
                          learning.rate = 1e-4, log.every = 5000,
                          architecture="large", batch.norm.layer = FALSE, dropout.layer = TRUE,
                          num.dim = 32, perplexity = 30, betas=0,
                          norm = TRUE, full.norm = FALSE,
                          early.stop = FALSE,
                          walker.loss = TRUE, reconc.loss = FALSE,
                          walker.weight = 1.0, classifier.weight = 1.0,
                          classifier.delay=NA, gpu.device = '0',
                          seed = 1234){

     valid_opts = c("steps", "steps.decoder", "batch.size", "learning.rate", "log.every", "architecture", "batch.norm.layer", "dropout.layer",
                   "num.dim", "perplexity", "betas", "norm", "full.norm", "early.stop", "walker.loss",
                   "reconc.loss", "walker.weight", "classifier.weight", "classifier.delay", "gpu.device", "seed")
     opts = data.frame(steps = steps,
                       steps.decoder=steps.decoder,
                       batch.size = batch.size,
                       learning.rate = learning.rate,
                       log.every = log.every,
                       architecture = architecture,
                       batch.norm.layer = batch.norm.layer,
                       dropout.layer = dropout.layer,
                       num.dim = num.dim,
                       perplexity = perplexity,
                       betas = betas,
                       norm = norm,
                       full.norm = full.norm,
                       early.stop = early.stop,
                       walker.loss = walker.loss,
                       reconc.loss = reconc.loss,
                       walker.weight = walker.weight,
                       classifier.weight = classifier.weight,
                       classifier.delay = classifier.delay,
                       gpu.device = as.character(gpu.device),
                       seed = seed,
                       stringsAsFactors=FALSE)
    colnames(opts) = valid_opts

    # if(!is.null(options)){
    #   if(all(names(options) %in% valid_opts)){
    #     ## Populate options with user supplied parameters
    #     for(name in names(options)){
    #       opts[,name] = options[[name]]
    #     }
    #   }else{ stop(paste0("These provided options are not valid: ", names(options)[which(!names(options) %in% valid_opts)])) }
    # }
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
