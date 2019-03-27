## Front end for scAlign that accepts matrix or Seurat input
#' Run scAlign framework
#'
#' Main function for scAlign that runs encoder and decoder networks
#'
#' @return scAlign class
#'
#' @param sce.object scAlign object.
#' @param options Training options for scAlign.
#' @param encoder.data Which data format to use for alignment.
#' @param decoder.data Which data format to use for interpolation.
#' @param supervised Run scAlign in supervised mode, requires labels.
#' @param run.encoder Run scAlign alignment procedure.
#' @param run.decoder Run scAlign projection through paired decoders.
#' @param device Specify hardware to use. May not work on all systems, manually set CUDA_VISIBLE_DEVICES if necessary.
#' @param log.dir Location to save results.
#' @param log.results Determines if results should be written to log.dir.
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
#' ## Build the SCE object for input to scAlign using Seurat preprocessing and variable gene selection
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
#' ## Run scAlign with high_var_genes
#' scAlignHSC = scAlign(scAlignHSC,
#'                    options=scAlignOptions(steps=1, log.every=1, norm=TRUE, early.stop=FALSE),
#'                    encoder.data="scale.data",
#'                    supervised='none',
#'                    run.encoder=TRUE,
#'                    run.decoder=FALSE,
#'                    log.dir=file.path('~/models','gene_input'),
#'                    device="CPU")
#'
#' @import SingleCellExperiment
#' @import tensorflow
#' @import methods
#'
#' @export
scAlign = function(sce.object,
                   options = scAlignOptions(),
                   encoder.data,
                   decoder.data=encoder.data,
                   ## Unified model options
                   supervised="none",
                   run.encoder=TRUE,
                   run.decoder=FALSE,
                   ## Where to save model results
                   log.dir="./models/",
                   log.results=FALSE,
                   ## Hardware selection
                   device="CPU"){

    ## Check tensorlfow install
    .check_tensorflow()

    ## Set options to class object
    metadata(sce.object)[["options"]] = options;

    ## Record arguments passed to this function
    metadata(sce.object)[["arguments"]] = scAlignArguments(sce.object,
                                                     encoder.data,
                                                     decoder.data,
                                                     supervised,
                                                     run.encoder,
                                                     run.decoder,
                                                     log.dir,
                                                     log.results,
                                                     device)

    ## Indepth argument sanity checking
    .check_all_args(sce.object)

    encoder_data = .data_setup(sce.object, encoder.data)
    object1 = encoder_data[[1]]; object2 = encoder_data[[2]];
    object1.name = encoder_data[[3]]; object2.name=encoder_data[[4]];
    data.use = encoder_data[[5]];

    ## Avoid multiple indexing
    options   = metadata(sce.object)[["options"]]
    arguments = metadata(sce.object)[["arguments"]]

    ## Set flags for all aspects of tensorflow training (encoder/decoder)
    FLAGS <- tensorflow::flags(
      ## Reporting/Verbosity ##
      tensorflow::flag_boolean('plot', TRUE, 'Investigate model usages'),
      tensorflow::flag_boolean('verbose', TRUE, 'Enable verbose runtime information'),
      tensorflow::flag_boolean('log.results', arguments$log.results, 'Save summary and graph information to log.dir'),
      tensorflow::flag_boolean('stack_trace', FALSE, 'Investigate model usages'),
      ## Saving ##
      tensorflow::flag_string('logdir', arguments$log.dir, 'Training log path.'),
      tensorflow::flag_integer('log_every_n_steps', options$log.every, 'Logging interval for training loop.'),
      tensorflow::flag_integer('max_checkpoints', 5, 'Maximum number of recent checkpoints to keep.'),
      tensorflow::flag_numeric('keep_checkpoint_every_n_hours', 5.0, 'How often checkpoints should be kept.'),
      tensorflow::flag_integer('save_summaries_secs', 300, 'How often should summaries be saved (in seconds).'),
      tensorflow::flag_integer('save_interval_secs', 600, 'How often should checkpoints be saved (in seconds).'),
      ## Early stopping ##
      tensorflow::flag_boolean('early_stopping', options$early.stop, 'Performing early stopping based on total loss value.'),
      ## Multi-GPU (currently unused) ##
      tensorflow::flag_string('master', '',  'BNS name of the TensorFlow master to use.'),
      tensorflow::flag_integer('ps_tasks', 0, 'The number of parameter servers. If the value is 0, then the parameters are handled locally by the worker.'),
      tensorflow::flag_integer('task', 0, 'The Task ID. This value is used when training with multiple workers to identify each worker.'),
      ## Encoder architecture ##
      tensorflow::flag_string('encoder', options$architecture, 'Which network architecture from architectures.py to use.'),
      tensorflow::flag_integer('emb_size', options$num.dim, 'Size of the embeddings to learn.'),
      tensorflow::flag_boolean('batch_norm', TRUE, 'To include batch_norm layers in model'),
      tensorflow::flag_numeric('batch_norm_decay', 0.99, 'Batch norm decay factor (unused at the moment'),
      ## Decoder architecture ##
      tensorflow::flag_string('decoder', options$architecture, 'Which network architecture from architectures.py to use for decoder network.'),
      ## Training ##
      tensorflow::flag_integer('unsup_batch_size', options$batch.size, 'Number of unlabeled samples per batch.'),
      tensorflow::flag_numeric('learning_rate', options$learning.rate, 'Initial learning rate.'),
      tensorflow::flag_numeric('minimum_learning_rate', 1e-8, 'Lower bound for learning rate.'),
      tensorflow::flag_numeric('decay_factor', 0.33, 'Learning rate decay factor.'),
      tensorflow::flag_numeric('decay_steps', floor((3/5)*options$steps), 'Learning rate decay interval in steps.'),
      tensorflow::flag_integer('max_steps', options$steps, 'Number of training steps.'),
      tensorflow::flag_integer('max_steps_decoder', options$steps, 'Number of training steps.'),
      tensorflow::flag_integer('random_seed', options$seed, 'Integer random seed used for labeled set selection.'),
      ## Loss function: walker loss for object1 ##
      tensorflow::flag_numeric('walker_weight', 1.0, 'Weight for walker loss.'),
      tensorflow::flag_string('walker_weight_envelope', 'None', 'Increase walker weight with an envelope: [None, sigmoid, linear]'),
      tensorflow::flag_integer('walker_weight_envelope_steps', 1, 'Number of steps (after delay), at which envelope saturates.'),
      tensorflow::flag_integer('walker_weight_envelope_delay', 1, 'Number of steps at which envelope starts.'),
      tensorflow::flag_integer('leaky_weights', 0, 'If true: original weighting growth used (leaky during classifier pre-training), else walker weight is restricted to exactly 0 before walker_weight_envelope_delay.'),
      ## Loss function: walker loss for object2 ##
      tensorflow::flag_numeric('target_walker_weight', 1.0, 'Weight for target walker loss.'),
      tensorflow::flag_string('target_walker_weight_envelope', 'None', 'Increase visit weight with an envelope: [None, sigmoid, linear]'),
      tensorflow::flag_integer('target_walker_weight_envelope_steps', 1, 'Number of steps (after delay) at which envelope saturates. -1 = follow walker loss env.'),
      tensorflow::flag_integer('target_walker_weight_envelope_delay', 1, 'Number of steps at which envelope starts. -1 = follow walker loss env.'),
      ## Loss function: classifier ##
      tensorflow::flag_string('supervised', arguments$supervised, 'What type of classifier to run during training'),
      tensorflow::flag_numeric('logit_weight', 1.0, 'Weight for logit loss.'),
      ## Loss function: visit (currently unused) ##
      tensorflow::flag_numeric('visit_weight', 0.0, 'Weight for visit loss.'),
      tensorflow::flag_string('visit_weight_envelope', 'None','Increase visit weight with an envelope: [None, sigmoid, linear]'),
      tensorflow::flag_integer('visit_weight_envelope_steps', 0, 'Number of steps (after delay) at which envelope saturates. -1 = follow walker loss env.'),
      tensorflow::flag_integer('visit_weight_envelope_delay', 0, 'Number of steps at which envelope starts. -1 = follow walker loss env.'),
      ## t-SNE kernel ##
      tensorflow::flag_string('kernel', 'tsne', 'Which kernel method tsne or original T (uniform),?'),
      tensorflow::flag_string('prob_comp', 'exp',  'How to compute p_ab, p_ba: non_exp (expless softmax), or softmax'),
      tensorflow::flag_string('self_sim', 'zero', 'To use self similarity in loss function, p_aba and T diagonal zero?'),
      tensorflow::flag_numeric('perplexity', options$perplexity, 'parameter used to control number of neighbors when computing T_ij'),
      tensorflow::flag_string('tsne_metric', 'euclidean', 'The metric to use when calculating distance between instances in a feature array.'),
      tensorflow::flag_string('tsne_method', 'exact', 'exact or approximately compute p_ij'),
      tensorflow::flag_string('tsne_init', 'random', 'If int, random_state is the seed used by the random number generator'),
      ## Data options ##
      tensorflow::flag_boolean('norm', options$norm, 'Perform L2 normalization during training on the mini batches.'),
      tensorflow::flag_boolean('full_norm', options$full.norm, 'Perform L2 normalization prior to training on the full data matrix.'),
      ## Testing options ##
      tensorflow::flag_integer('mini_batch', 50, 'Number samples per testing batch.'),
      ## Hardware ##
      tensorflow::flag_string('cuda_device', options$gpu.device, 'Select the GPU for this job'))

    ## Verbosity of tensorflow output. Filters: (1 INFO) (2 WARNINGS) (3 ERRORS)
    Sys.setenv(TF_CPP_MIN_LOG_LEVEL=3);
    Sys.setenv(CUDA_VISIBLE_DEVICES=options$gpu.device);

    ## Hardware configurations for GPU if enabled
    config = tf$ConfigProto(gpu_options = tf$GPUOptions(allow_growth=TRUE),
                            allow_soft_placement=TRUE,
                            log_device_placement=FALSE,
                            device_count = dict('GPU', 1))

    if(FLAGS$log.results == TRUE){
      dir.create(file.path(FLAGS$logdir, '/model_full'), showWarnings = FALSE)
      dir.create(file.path(FLAGS$logdir, '/model_full/plots'), showWarnings = FALSE)
      for(itr in seq_len(length(unique(colData(sce.object)[,"group.by"])))){
          dir.create(file.path(FLAGS$logdir, paste0(unique(colData(sce.object)[,"group.by"])[itr], '_decoder')),
                     showWarnings = FALSE)
      }
      ## Write out all run options for reproducability
      write.table(as.data.frame(FLAGS), file=file.path(FLAGS$logdir, 'model_full/run_flags.txt'), sep="\t", row.names=FALSE, col.names=TRUE)
    }

    ## Supervised options
    src_mode = ifelse(is.element(FLAGS$supervised, c(object1.name, "both")), "supervised", "unsupervised")
    trg_mode = ifelse(is.element(FLAGS$supervised, c(object2.name, "both")), "supervised", "unsupervised")

    ## Determine data shape and label space once
    num_labels = length(unique(colData(sce.object)[,"scAlign.labels"])) ## determines number of logits for classifier
    data_shape = ncol(object1) ## source and target should have same shape

    ############################################################################
    ## Run scAlign
    ############################################################################
    tryCatch({
      ## Domain Adaptation
      if(run.encoder == TRUE){
        print("============== Step 1/3: Encoder training ===============")
        aligned = encoderModel_train_encoder(FLAGS, 'alignment', config,
                                             num_labels, data_shape,
                                             object1.name, object2.name,
                                             object1, as.integer(
                                                        as.factor(colData(sce.object)[,"scAlign.labels"][object1.name == colData(sce.object)[,"group.by"]])),
                                             object2, as.integer(
                                                        as.factor(colData(sce.object)[,"scAlign.labels"][object2.name == colData(sce.object)[,"group.by"]])))
        reducedDim(sce.object, paste0("ALIGNED-", data.use)) = aligned[[1]]
        metadata(sce.object)[[paste0("LOSS-", data.use)]] = aligned[[2]]
      }
    }, error=function(e){
      print("Error during alignment, returning scAlign class.")
      print(e)
      return(sce.object)
    })

    tryCatch({
      ## Projection
      if(run.decoder == TRUE){

        ## Try to load aligned data
        if(run.encoder == TRUE){
          emb_dataset = reducedDim(sce.object, type=paste0("ALIGNED-", data.use))
        }else{
          emb_dataset = tryCatch({
              as.matrix(read.csv(paste0(FLAGS$logdir, '/model_alignment/train/emb_activations_', FLAGS$max_steps, '.csv'), header=FALSE, stringsAsFactors=FALSE))
          }, error=function(e) {
               stop("Error loading previously aligned data, does not exist.")
          })
        }

        data_embed_source = emb_dataset[seq_len(nrow(object1)),]
        data_embed_target = emb_dataset[(nrow(object1)+1):(nrow(object1)+nrow(object2)),]

        ## Get decoder data
        decoder_data = .data_setup(sce.object, decoder.data)
        object1 = decoder_data[[1]]; object2 = decoder_data[[2]];
        object1.name = decoder_data[[3]]; object2.name = decoder_data[[4]];
        data.use = decoder_data[[5]];

        ## Train models
        print(paste0("============== Step 2/3: ", object1.name ," decoder training ==============="))
        projected = decoderModel_train_decoder(FLAGS, config, object1.name,
                                                       object1,
                                                       data_embed_source, emb_dataset)

        reducedDim(sce.object, paste0(object1.name, "2", object2.name)) = projected

        print(paste0("============== Step 3/3: ", object2.name ," decoder training ==============="))
        projected = decoderModel_train_decoder(FLAGS, config, object2.name,
                                                       object2,
                                                       data_embed_target, emb_dataset)

        reducedDim(sce.object, paste0(object2.name, "2", object1.name)) = projected
      }
    }, error=function(e){
      print("Error during interpolation, returning scAlign class.")
      print(e)
      return(sce.object)
    })

    return(sce.object) ## Return Class
}
