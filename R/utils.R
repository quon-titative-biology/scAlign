#' Creates tsne plot
#'
#' Helper function to plot aligned data from the results of running scAlign().
#'
#' @return ggplot2 object
#'
#' @param object scAlign class object with aligned data
#' @param data.use Specifies which alignment results to use.
#' @param labels.use Specifies "dataset" or "celltype" labeling from object meta.data.
#' @param labels Object labels
#' @param cols Colours for plot
#' @param title ggplot title
#' @param point.size Size of geom_point
#' @param legend Determines if legend should be drawn
#' @param seed Random seed for reproducability
#' @param ... Additional arguments to Rtsne function
#'
#' @import Rtsne
#' @import ggplot2
#'
#' @examples
#'
#'  library(SingleCellExperiment)
#'
#'  ## Input data, 1000 genes x 100 cells
#'  data = matrix( rnorm(1000*100,mean=0,sd=1), 1000, 100)
#'  rownames(data) = paste0("gene", seq_len(1000))
#'  colnames(data) = paste0("cell", seq_len(100))
#'
#'  age    = c(rep("young",50), rep("old",50))
#'  labels = c(c(rep("type1",25), rep("type2",25)), c(rep("type1",25), rep("type2",25)))
#'
#'  ctrl.data = data[,which(age == "young")]
#'  stim.data = data[,which(age == "old")]
#'
#'  ## Build the SCE object for input to scAlign using Seurat preprocessing and variable gene selection
#'  ctrlSCE <- SingleCellExperiment(
#'                assays = list(scale.data = data[,which(age == "young")]))
#'
#'  stimSCE <- SingleCellExperiment(
#'                assays = list(scale.data = data[,which(age == "old")]))
#'
#'  ## Build the scAlign class object and compute PCs
#'  scAlignHSC = scAlignCreateObject(sce.objects = list("YOUNG"=ctrlSCE,
#'                                                      "OLD"=stimSCE),
#'                                   labels = list(labels[which(age == "young")],
#'                                                 labels[which(age == "old")]),
#'                                   pca.reduce = FALSE,
#'                                   cca.reduce = FALSE,
#'                                   project.name = "scAlign_Kowalcyzk_HSC")
#'
#'  ## Run scAlign with high_var_genes
#'  scAlignHSC = scAlign(scAlignHSC,
#'                     options=scAlignOptions(steps=100,
#'                                            log.every=100,
#'                                            norm=TRUE,
#'                                            early.stop=FALSE),
#'                     encoder.data="scale.data",
#'                     supervised='none',
#'                     run.encoder=TRUE,
#'                     run.decoder=FALSE,
#'                     log.results=FALSE,
#'                     log.dir=file.path('~/models','gene_input'),
#'                     device="CPU")
#'
#'  ## Plot alignment for 3 input types
#'  example_plot = PlotTSNE(scAlignHSC,
#'                            "ALIGNED-GENE",
#'                            "scAlign.labels",
#'                            title="scAlign-Gene",
#'                            perplexity=30)
#'
#' @export
PlotTSNE = function(object, data.use, labels.use="scAlign.labels", cols=NULL, title="", legend="none", point.size=3, seed=1234, ...){
    x=y=NULL ## Appease R checker, doesn't like ggplot2 aes() variables
    tryCatch({
      if(data.use %in% names(assays(object))){
        res = Rtsne(assay(object, data.use), ...)
      }else if(data.use %in% names(reducedDims(object))){
        res = Rtsne(reducedDim(object, data.use), ...)
      }else{
        print("Data type not defined in assays or reducedDims")
        return()
      }
      labels = as.character(colData(object)[,labels.use])
      plot.me <- data.frame(x=res$Y[,1], y=res$Y[,2], labels=labels, stringsAsFactors=FALSE)
      tsne.plot <- ggplot(plot.me, aes(x=x, y=y, colour = labels))
      if(!is.null(cols)){ tsne.plot <- tsne.plot + scale_colour_manual(values=cols) }
      tsne.plot <- tsne.plot +
                       geom_point(size=point.size) +
                       xlab('t-SNE 1') +
                       ylab('t-SNE 2') +
                       ggtitle(title) +
                       theme_bw()      +
                       theme(panel.border = element_blank(),
                           panel.grid.major = element_blank(),
                           panel.grid.minor = element_blank(),
                           panel.background = element_rect(fill = "transparent"), # bg of the panel
                           plot.background = element_rect(fill = "transparent", color = NA),
                           axis.line = element_line(colour = 'black',size=1),
                           plot.title = element_text(color='black', size=20, hjust = 0.5),
                           axis.text.x = element_text(color='black', size=14),
                           axis.text.y = element_text(color='black', size=14),
                           axis.title=element_text(size=24),
                           legend.position=legend,
                           legend.background = element_rect(fill = "transparent"), # get rid of legend bg
                           legend.box.background = element_rect(fill = "transparent"), # get rid of legend panel bg
                           legend.title=element_blank(),
                           legend.text=element_text(size=rel(3.0)))
      return(tsne.plot)
    }, error = function(e){
      print("Error plotting the data provided.")
      print(e)
      return(NULL)
    })
}

#' Computes gaussian kernel matrix
#'
#' Tensorflow implementation of tSNE's gaussian kernel.
#'
#' @return Tensorflow op
#'
#' @param data cell x feature data matrix
#' @param data_shape number of features for data
#' @param labels cell x 1 annotation (label) vector
#' @param method Kernel to compute pairwise cell similarity
#' @param perplexity neighborhood parameter for gaussian kernel
#' @param diag indicator for self similarity
#'
#' @examples
#'
#' ## Input data, 100 cells x 10 features
#' data = matrix(sample.int(1000, 100*10, TRUE), 100, 10)
#' rownames(data) = paste0("cell", seq_len(100))
#' colnames(data) = paste0("gene", seq_len(10))
#'
#' result = gaussianKernel(data, nrow(data))
#'
#' @import tensorflow
#'
#' @export
gaussianKernel = function(data, data_shape, labels=NULL, method=NULL, perplexity=30, diag="zero"){
  ## Hardware configurations for GPU if enabled
  config = tf$ConfigProto(gpu_options = tf$GPUOptions(allow_growth=TRUE),
                          allow_soft_placement=TRUE,
                          log_device_placement=FALSE,
                          device_count = dict('GPU', 1))
  sess = NULL ## Appease R check
  with(tf$Session(config=config) %as% sess, {
    data=tf$cast(data, tf$float64)
    data = tf$nn$l2_normalize(data, axis=as.integer(1))
    ## Defines the similarity matrix T used for asssociation loss
    kernel = encoderModel_gaussian_kernel(data, dim=data_shape, perplexity=perplexity, diag=diag)
    ## Cast down for consistent data type
    return(sess$run(tf$cast(kernel, tf$float32)))
  })
}

#' Check for whole number
#'
#' @return Boolean
#'
#' @keywords internal
.is.wholenumber = function(x, tol=.Machine$double.eps^0.5){abs(x - round(x)) < tol}

#' Creates tsne plot
#'
#' @return ggplot2 object
#'
#' @param data Current aligned embeddings
#' @param labels Object labels
#' @param file_out File name to save plot
#'
#' @import Rtsne
#' @import ggplot2
#'
#' @keywords internal
.plotTSNE = function(data, labels, file_out="~/scAlign_default_plot.png"){
    x=y=NULL ## Appease R checker, doesn't like ggplot2 aes() variables
    res = Rtsne(data, PCA=FALSE, verbose=FALSE)
    plot.me <- data.frame(x=res$Y[,1], y=res$Y[,2], labels=labels, stringsAsFactors=FALSE)
    tsne.plot <- ggplot(plot.me, aes(x=x, y=y, colour = labels), size=5) +
                     geom_point() +
                     xlab('t-SNE 1') +
                     ylab('t-SNE 2') +
                     theme_bw()      +
                     theme(panel.border = element_blank(),
                         panel.grid.major = element_blank(),
                         panel.grid.minor = element_blank(),
                         panel.background = element_rect(fill = "transparent"), # bg of the panel
                         plot.background = element_rect(fill = "transparent", color = NA),
                         axis.line = element_line(colour = 'black',size=1),
                         plot.title = element_text(color='black', size=32),
                         axis.text.x = element_text(color='black', size=14),
                         axis.text.y = element_text(color='black', size=14),
                         axis.title=element_text(size=24),
                         legend.position="right",
                         legend.background = element_rect(fill = "transparent"), # get rid of legend bg
                         legend.box.background = element_rect(fill = "transparent"), # get rid of legend panel bg
                         legend.title=element_blank(),
                         legend.text=element_text(size=rel(3.0)))
    ggsave(tsne.plot, filename=file_out, width=14, height=12)
}

#' Computes alignment score
#'
#' @return Boolean
#'
#' @import FNN
#'
#' @keywords internal
alignment_score <- function(data, source_labels, target_labels, nn=0){
  dist = c(rep("source", length(source_labels)), rep("target", length(target_labels)))
  if(nn == 0){nn = ceiling(nrow(data) * 0.01 * 2)}
  object.fnn <- get.knn(data, k=nn)
  alignment.score = sapply(seq_len(nrow(data)), function(x) {
    length(which(dist[object.fnn$nn.index[x, ]] == dist[x]))
  })
  alignment.score = 1-((mean(alignment.score)-nn/2)/(nn-nn/2))
  return(alignment.score)
}

#' Check arguments
#'
#' @return Nothing
#'
#' @keywords internal
.check_all_args = function(sce.object){

  ## Avoid multiple indexing
  options   = metadata(sce.object)[["options"]]
  arguments = metadata(sce.object)[["arguments"]]

  ## Ensure data to be used for encoder is available
  for(element in arguments[,"encoder.data"]){
    if((is.element(element, names(assays(sce.object))) ||
        is.element(element, names(reducedDims(sce.object)))) == FALSE){
          stop(paste0(element, " is not reachable in assays or reducedDims slots of combined SCE object."))
        }
  }
  ## Ensure data to be used for decoder is available
  for(element in arguments[,"decoder.data"]){
    if((is.element(element, names(assays(sce.object))) ||
        is.element(element, names(reducedDims(sce.object)))) == FALSE){
          stop(paste0(element, " is not reachable in assays or reducedDims slots of combined SCE object."))
        }
  }

  ## Parameter check
  if (!is.character(arguments[,"supervised"]) || !is.element(arguments[,"supervised"], c("none", unique(sce.object$group.by), "both"))) { stop("supervised should be \"none\", \"[object.name(s)]\" or \"both\".") }
  if (!is.character(arguments[,"device"])) { stop("device should be a string.") }
  if (!.is.wholenumber(options[,"batch.size"])) { stop("batch.size should be an integer.") }
  if (!.is.wholenumber(options[,"perplexity"])) { stop("perplexity should be an integer.") }
  if (!is.numeric(options[,"learning.rate"]) || as.numeric(options[,"learning.rate"])<=0.0) { stop("Incorrect learning.rate scale.") }
  if (!(is.logical(options[,"norm"]))) { stop("norm should be TRUE or FALSE.") }
  if (!(is.logical(arguments[,"run.encoder"]))) { stop("run.encoder should be TRUE or FALSE") }
  if (!(is.logical(arguments[,"run.decoder"]))) { stop("run.decoder should be TRUE or FALSE") }

  ## Ensure results directories can be created
  tryCatch({
    dir.create(file.path(arguments[,"log.dir"]), recursive=TRUE, showWarnings = TRUE)
  }, warning=function(w){

  }, error=function(e){
    stop("Invalid path for log.dir")
  })

  if(options[,"batch.size"] <= 0){
    stop("Incorrect minibatch_size, must not be greater than combined number of samples or less than 1.")
  }

  if(options[,"steps"] < 100){
    print("Training steps should be at least 100.")
  }

  if(options[,"log.every"] > options[,"steps"]){
    stop("log_results_every_n_steps should not be larger than training_steps.")
  }

  if(options[,"num.dim"] <= 0){
    stop("Embedding_size should be greater than 0.")
  }

  tryCatch({
      match.fun(paste0("encoder_", options$architecture))
  }, error=function(e) {
      stop("architecture_encoder does not exist in architectures.R")
  })

  tryCatch({
      match.fun(paste0("decoder_", options$architecture))
  }, error=function(e) {
      stop("architecture_decoder does not exist in architectures.R")
  })
}

#' Check tf install
#'
#' @return Nothing
#'
#' @keywords internal
.check_tensorflow = function(){
  sess = NULL ## Appease R check
  tryCatch({
    with(tf$Session() %as% sess, {
      sess$run(tf$Print("", list("Passed"), "TensorFlow check: "))
    })
  }, error=function(e) {
      stop("Error with system install of tensorflow, check R for Tensorflow docs.")
  })
}

#' Perform early stopping
#'
#' @return Earlying stopping flag
#'
#' @keywords internal
.early_stop = function(loss_tracker, step, patience_count, early_stopping_active, patience=30, min_delta=0.01){
  ## Early stopping enabled steps exceeds patience
  if(step > early_stopping_active){
    loss_avg = mean(loss_tracker[(length(loss_tracker)-50):(length(loss_tracker)-1)])
    if(abs((loss_avg - loss_tracker[length(loss_tracker)])/((loss_avg + loss_tracker[length(loss_tracker)])/2.0)) > min_delta){
      patience_count = 0
    }else{
      patience_count = patience_count + 1
    }
  }
  if(patience_count >= patience){
    patience_count = TRUE;
  }
  return(patience_count)
}

#' Sets the learning rate for optimizer
#'
#' @return Tensorflow op
#'
#' @keywords internal
.learning_rate = function(step, decay_step, FLAGS){
    return(tf$maximum(
            tf$train$exponential_decay(
               FLAGS$learning_rate,
               step,
               decay_step,
               FLAGS$decay_factor),
           FLAGS$minimum_learning_rate))
}

#' Sets data
#'
#' @return Extracted data
#'
#' @keywords internal
.data_setup = function(sce.object, data.use){
  ## Get dataset names
  object1.name = unique(colData(sce.object)[,"group.by"])[1]
  object2.name = unique(colData(sce.object)[,"group.by"])[2]
  ## Get indices of two datasets (using overloaded ==)
  object1.idx = object1.name == colData(sce.object)[,"group.by"]
  object2.idx = object2.name == colData(sce.object)[,"group.by"]
  ## Set the data for current alignment
  if(is.element(data.use, names(assays(sce.object)))){
    ## placeholders, works for now (data) until work on multi alignment
    object1 = t(as.matrix(assay(sce.object, data.use))[,object1.idx]);
    object2 = t(as.matrix(assay(sce.object, data.use))[,object2.idx]);
    data.use = "GENE"
  }else if(is.element(data.use, names(reducedDims(sce.object)))){
    ## placeholders, works for now (data) until work on multi alignment
    object1 = as.matrix(reducedDim(sce.object, data.use)[object1.idx,]);
    object2 = as.matrix(reducedDim(sce.object, data.use)[object2.idx,]);
  }else{
    print("Choice for data.use does not exist in SCE assays or reducedDims.")
  }
  return(list(object1, object2, object1.name, object2.name, data.use))
}
