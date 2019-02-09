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
#' ## Run scAlign with high_var_genes
#' scAlignHSC = scAlign(scAlignHSC,
#'                      options=scAlignOptions(steps=1, log.every=1, early.stop=FALSE, architecture="large"),
#'                      encoder.data="scale.data",
#'                      supervised='none',
#'                      run.encoder=TRUE,
#'                      run.decoder=FALSE,
#'                      log.results=FALSE,
#'                      device="CPU")
#'
#' ## Plot alignment for 3 input types
#' example_plot = PlotTSNE(scAlignHSC, "GENE", "labels", title="scAlign-Gene", perplexity=30)
#'
#' @export
PlotTSNE = function(object, data.use, labels.use, cols=NULL, title="", legend="none", seed=1234, ...){
    x=y=NULL ## Appease R checker, doesn't like ggplot2 aes() variables
    res = Rtsne(object@aligned.data[[data.use]]@embedding, ...)
    labels = object@meta.data[,labels.use]
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
.check_all_args = function(object){

  ## Parameter check
  if (!is.character(object@arguments[,"supervised"]) || !is.element(object@arguments[,"supervised"], c("none", names(object@scale.data), "both"))) { stop("supervised should be \"none\", \"[object.name(s)]\" or \"both\".") }
  if (!is.character(object@arguments[,"device"])) { stop("device should be a string.") }
  if (!.is.wholenumber(object@options[,"batch.size"])) { stop("batch.size should be an integer.") }
  if (!.is.wholenumber(object@options[,"perplexity"])) { stop("perplexity should be an integer.") }
  if (!is.numeric(object@options[,"learning.rate"]) || as.numeric(object@options[,"learning.rate"])<=0.0) { stop("Incorrect learning.rate scale.") }
  if (!(is.logical(object@options[,"norm"]))) { stop("norm should be TRUE or FALSE.") }
  if (!(is.logical(object@arguments[,"run.encoder"]))) { stop("run.encoder should be TRUE or FALSE") }
  if (!(is.logical(object@arguments[,"run.decoder"]))) { stop("run.decoder should be TRUE or FALSE") }

  ## Ensure results directories can be created
  tryCatch({
    dir.create(file.path(object@arguments[,"log.dir"]), recursive=TRUE, showWarnings = TRUE)
  }, warning=function(w){

  }, error=function(e){
    stop("Invalid path for log.dir")
  })

  if(object@options[,"batch.size"] <= 0){
    stop("Incorrect minibatch_size, must not be greater than combined number of samples or less than 1.")
  }

  if(object@options[,"steps"] < 1000){
    print("Training steps should be at least 1000.")
  }

  if(object@options[,"log.every"] > object@options[,"steps"]){
    stop("log_results_every_n_steps should not be larger than training_steps.")
  }

  if(object@options[,"num.dim"] <= 0){
    stop("Embedding_size should be greater than 0.")
  }

  tryCatch({
      match.fun(paste0("encoder_", object@options$architecture))
  }, error=function(e) {
      stop("architecture_encoder does not exist in architectures.R")
  })

  tryCatch({
      match.fun(paste0("decoder_", object@options$architecture))
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
.data_setup = function(object, data.use){
  ## Set the data for current alignment
  if(is.element(data.use, c("raw.data", "scale.data", names(object@reduced.data)))){
    if(data.use == "raw.data"){
      ## placeholders, works for now (data) until work on multi alignment
      object1 = t(object@raw.data[[1]]); object1.name = names(object@raw.data)[1]
      object2 = t(object@raw.data[[2]]); object2.name = names(object@raw.data)[2]
      data.use = "GENE"
    } else if(data.use == "scale.data"){
      ## placeholders, works for now (data) until work on multi alignment
      object1 = t(object@scale.data[[1]]); object1.name = names(object@scale.data)[1]
      object2 = t(object@scale.data[[2]]); object2.name = names(object@scale.data)[2]
      data.use = "GENE"
    } else{
      ## Extract object1 data
      object1.name = names(object@scale.data)[1]
      object1 = object@reduced.data[[data.use]]@embedding[which(object@meta.data$group == object1.name),];
      ## Extract object2 data
      object2.name = names(object@scale.data)[2]
      object2 = object@reduced.data[[data.use]]@embedding[which(object@meta.data$group == object2.name),];
    }
  }else{
    print("Choice for data.use does not exist in scAlign class.")
  }
  return(list(object1, object2, object1.name, object2.name, data.use))
}
