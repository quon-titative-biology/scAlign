#' Creates tsne plot
#'
#' @return ggplot2 object
#'
#' @param object scAlign class object with aligned data
#' @param data.use Specifies which alignment results to use.
#' @param labels.use Specifies "dataset" or "celltype" labeling from object meta.data.
#' @param labels Object labels
#' @param cols Colours for plot
#' @param title ggplot title
#' @param legend Determines if legend should be drawn
#' @param seed Random seed for reproducability
#' @param ... Additional arguments to Rtsne function
#'
#' @import Rtsne
#' @import ggplot2
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
#' ctrl@meta.data$stim  = "YOUNG"
#' ctrl@meta.data$label = labels[which(age == "young")]
#' ctrl = ScaleData(ctrl, do.scale=TRUE, do.center=TRUE, scale.max=50, display.progress = TRUE)
#'
#' stim = CreateSeuratObject(raw.data = stim.data, project = "MOUSE_AGE", min.cells = 0)
#' stim@meta.data$stim = "OLD"
#' stim@meta.data$label = labels[which(age == "old")]
#' stim = ScaleData(stim, do.scale=TRUE, do.center=TRUE, scale.max=50, display.progress = TRUE)
#'
#' ## Build the SCE object for input to scAlign using Seurat preprocessing and variable gene selection
#' ctrl.sce <- SingleCellExperiment(
#'               assays = list(
#'                 counts = ctrl@raw.data,
#'                 scale.data = ctrl@scale.data),
#'               colData = ctrl@meta.data)
#'
#' stim.sce <- SingleCellExperiment(
#'               assays = list(
#'                 counts = stim@raw.data,
#'                 scale.data = stim@scale.data),
#'               colData = stim@meta.data)
#'
#' ## Build the scAlign class object and compute PCs
#' scAlignHSC = scAlignCreateObject(sce.objects = list("YOUNG"=ctrl.sce,
#'                                                     "OLD"=stim.sce),
#'                                  labels = list(ctrl.sce@colData@listData$label,
#'                                                stim.sce@colData@listData$label),
#'                                  pca.reduce = TRUE,
#'                                  pcs.compute = 50,
#'                                  cca.reduce = TRUE,
#'                                  ccs.compute = 15,
#'                                  project.name = "scAlign_Kowalcyzk_HSC")
#'
#' ## Run scAlign with high_var_genes
#' scAlignHSC = scAlign(scAlignHSC,
#'                     options=scAlignOptions(steps=100, log.every=100, early.stop=FALSE, architecture="small"),
#'                     encoder.data="scale.data",
#'                      supervised='none',
#'                      run.encoder=TRUE,
#'                      run.decoder=TRUE,
#'                      log.results=FALSE,
#'                      device="CPU")

#' print(scAlignHSC@reducedDims[["ALIGNED-GENE"]])
#'
#' ## Plot alignment for 3 input types
#' example_plot = PlotTSNE(scAlignHSC, "ALIGNED-GENE", "labels", title="scAlign-Gene", perplexity=30)
#'
#' @export
PlotTSNE = function(object, data.use, labels.use, cols=NULL, title="", legend="none", seed=1234, ...){
    x=y=NULL ## Appease R checker, doesn't like ggplot2 aes() variables
    tryCatch({
      res = Rtsne(object@reducedDims[[data.use]], ...)
      if(labels.use == "labels"){
        labels = object@colData@listData[["scAlign.labels"]]
      }else{
        labels = object@colData@listData[["group.by"]]
      }
      plot.me <- data.frame(x=res$Y[,1], y=res$Y[,2], labels=labels, stringsAsFactors=FALSE)
      tsne.plot <- ggplot(plot.me, aes(x=x, y=y, colour = labels))
      if(!is.null(cols)){ tsne.plot <- tsne.plot + scale_colour_manual(values=cols) }
      tsne.plot <- tsne.plot +
                       geom_point() +
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
                           plot.title = element_text(color='black', size=64, hjust = 0.5),
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

#' Check arguments
#'
#' @return Nothing
#'
#' @keywords internal
.check_all_args = function(sce.object){

  arguments = sce.object@metadata[["arguments"]]
  options   = sce.object@metadata[["options"]]

  ## Parameter check
  if (!is.character(arguments[,"supervised"]) || !is.element(arguments[,"supervised"], c("none", names(sce.object@assays), "both"))) { stop("supervised should be \"none\", \"[object.name(s)]\" or \"both\".") }
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

  if(options[,"steps"] < 1000){
    print("Training steps should be at least 1000.")
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
    loss_avg = mean(loss_tracker[(length(loss_tracker)-100):(length(loss_tracker)-1)])
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
               FLAGS$decay_factor,
               staircase=TRUE),
           FLAGS$minimum_learning_rate))
}

#' Sets data
#'
#' @return Extracted data
#'
#' @keywords internal
.data_setup = function(sce.object, data.use){
  ## Get dataset names
  object1.name = unique(sce.object@colData@listData[["group.by"]])[1]
  object2.name = unique(sce.object@colData@listData[["group.by"]])[2]
  ## Get indices of two datasets (using overloaded ==)
  object1.idx = object1.name == sce.object@colData@listData[["group.by"]]
  object2.idx = object2.name == sce.object@colData@listData[["group.by"]]
  ## Set the data for current alignment
  if(is.element(data.use, names(sce.object@assays))){
    ## placeholders, works for now (data) until work on multi alignment
    object1 = t(sce.object@assays[[data.use]][,object1.idx]);
    object2 = t(sce.object@assays[[data.use]][,object2.idx]);
    data.use = "GENE"
  }else if(is.element(data.use, names(sce.object@reducedDims))){
    ## placeholders, works for now (data) until work on multi alignment
    object1 = t(sce.object@reducedDims[[data.use]][,object1.idx]);
    object2 = t(sce.object@reducedDims[[data.use]][,object2.idx]);
  }else{
    print("Choice for data.use does not exist in SCE assays or reducedDims.")
  }
  return(list(object1, object2, object1.name, object2.name, data.use))
}
