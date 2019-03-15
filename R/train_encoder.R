#' define kernel matrix
#'
#' Adds the defined kernel to the operation graph for high dimensional pairwise data similarity
#'
#' @return Tensorflow op
#'
#' @param method Kernel to compute pairwise cell similarity
#' @param data cell x feature data matrix
#' @param labels cell x 1 annotation (label) vector
#' @param data_shape number of features for data
#' @param perplexity neighborhood parameter for gaussian kernel
#' @param diag indicator for self similarity
#'
#' @import tensorflow
#' @import tfdatasets
#'
#' @keywords internal
define_kernel_matrix = function(method, data=NULL, labels=NULL, data_shape, perplexity=NULL, diag="zero"){
    ## Defines the similarity matrix T used for asssociation loss
    kernel = encoderModel_gaussian_kernel(data=tf$cast(data, tf$float64), dimen=data_shape, perplexity=perplexity, diag=diag)
    ## Cast down for consistent data type
    return(tf$cast(kernel, tf$float32))
}

#' encoder for scAlign
#'
#' Aligns paired data into a joint embedding space.
#'
#' @return Final alignment results
#'
#' @param FLAGS Tensorflow run arguments.
#' @param CV Model that is being run.
#' @param config Hardware configuration options.
#' @param num_labels Number of unique labels.
#' @param data_shape Dimensions of input data.
#' @param source_data Source data matrix (cell x feature).
#' @param source_labels Source data labels.
#' @param target_data Target data matrix (cell x feature).
#' @param target_labels Target data labels.
#'
#' @import tensorflow
#' @import purrr
#'
#' @keywords internal
encoderModel_train_encoder = function(FLAGS, CV, config,
                                      num_labels, data_shape,
                                      obj1_name, obj2_name,
                                      source_data, source_labels,
                                      target_data, target_labels){

  ## Define network structure
  graph = tf$Graph()
  with(graph$as_default(), {
    print("Graph construction")
    ## Set up inputs. Mini-batch via placeholder and dataset iterators
    with(tf$name_scope("source_data"), {
      if(FLAGS$supervised == "source" | FLAGS$supervised == "both"){
        source_data_ph   = tf$placeholder(tf$float32, shape(NULL,data_shape))
        source_labels_ph = tf$placeholder(tf$uint8, shape(NULL))
        source_dataset = tf$data$Dataset$from_tensor_slices(tuple(source_data_ph, source_labels_ph))
      }else{
        source_data_ph   = tf$placeholder(tf$float32, shape(NULL,data_shape))
        source_dataset = tf$data$Dataset$from_tensor_slices(source_data_ph)
      }
      source_dataset = source_dataset$shuffle(dim(source_data)[1])
      source_dataset = source_dataset$`repeat`()
      source_dataset = source_dataset$batch(FLAGS$unsup_batch_size)
      source_iter    = source_dataset$make_initializable_iterator()
      source_batch   = source_iter$get_next()
      if(typeof(source_batch) != "list"){source_batch = list(source_batch)}
    })

    with(tf$name_scope("target_data"), {
      if(FLAGS$supervised == "target" | FLAGS$supervised == "both"){
        target_data_ph   = tf$placeholder(tf$float32, shape(NULL,data_shape))
        target_labels_ph = tf$placeholder(tf$uint8, shape(NULL))
        target_dataset = tf$data$Dataset$from_tensor_slices(tuple(target_data_ph, target_labels_ph))
      }else{
        target_data_ph   = tf$placeholder(tf$float32, shape(NULL,data_shape))
        target_dataset = tf$data$Dataset$from_tensor_slices(target_data_ph)
      }
      target_dataset = target_dataset$shuffle(dim(target_data)[1])
      target_dataset = target_dataset$`repeat`()
      target_dataset = target_dataset$batch(FLAGS$unsup_batch_size)
      target_iter    = target_dataset$make_initializable_iterator()
      target_batch   = target_iter$get_next()
      if(typeof(target_batch) != "list"){target_batch = list(target_batch)}
    })

    # Create function that defines the network.
    model_function = partial(
        match.fun(paste0("encoder_", FLAGS$encoder)),
        input_size=data_shape,
        complexity=3,
        emb_size=FLAGS$emb_size,
        batch_norm=FLAGS$batch_norm)

    ## normalize per batch
    if(FLAGS$norm == TRUE){
      source_batch[[1]] = tf$nn$l2_normalize(source_batch[[1]], axis=as.integer(1))
      target_batch[[1]] = tf$nn$l2_normalize(target_batch[[1]], axis=as.integer(1))
    }

    ## Define test first, also acts as network initializer.
    ## tf.get_variable() is only called when reuse=FALSE for named/var scopes...
    test_in = tf$placeholder(tf$float32, shape(NULL,data_shape), 'test_in')
    test_emb = encoderModel_data_to_embedding(model_function, test_in, is_training=FALSE)
    test_logit = encoderModel_embedding_to_logit(test_emb, num_labels, is_training=FALSE)

    source_emb = encoderModel_data_to_embedding(model_function, source_batch[[1]], is_training=TRUE)
    target_emb = encoderModel_data_to_embedding(model_function, target_batch[[1]], is_training=TRUE)

    ## Add source assoc loss
    if(FLAGS$walker_weight != 0.0){
        print("Adding source walker loss")
        ## Specific form of walker loss
        T_source = define_kernel_matrix(method=FLAGS$kernel, data=source_batch[[1]], data_shape=FLAGS$unsup_batch_size, perplexity=FLAGS$perplexity)
        encoderModel_add_semisup_loss_data_driven(T_source,
                                                   source_emb,
                                                   target_emb,
                                                   FLAGS$unsup_batch_size, ## Replace for balanced case
                                                   walker_weight=FLAGS$walker_weight,
                                                   mode='source',
                                                   comp=FLAGS$prob_comp,
                                                   debug=FLAGS$verbose)
    }else{print("SOURCE STRUCUTRE NOT ENFORCED")}

    ## Add target assoc loss
    if(FLAGS$target_walker_weight != 0.0){
        print("Adding target walker loss")
        T_target = define_kernel_matrix(method=FLAGS$kernel, data=target_batch[[1]], data_shape=FLAGS$unsup_batch_size, perplexity=FLAGS$perplexity)
        encoderModel_add_semisup_loss_data_driven(T_target,
                                                   target_emb,
                                                   source_emb,
                                                   FLAGS$unsup_batch_size, ## Replace for balanced case
                                                   walker_weight=FLAGS$target_walker_weight,
                                                   mode='target',
                                                   comp=FLAGS$prob_comp,
                                                   debug=FLAGS$verbose)
    }else{print("TARGET STRUCUTRE NOT ENFORCED")}

    ## Add classificaiton loss
    if(FLAGS$supervised != 'none'){
        print("Adding classifier loss ")
        ## source specific loss for classifier
        if(FLAGS$supervised == obj1_name){
            ## Compute unnormalized class probabilities
            logits = encoderModel_embedding_to_logit(source_emb, num_labels)
            encoderModel_add_logit_loss(logits,
                           source_batch[[2]],
                           weight=FLAGS$logit_weight)
        }else if(FLAGS$supervised == obj2_name){
            ## Compute unnormalized class probabilities
            logits = encoderModel_embedding_to_logit(target_emb, num_labels)
            encoderModel_add_logit_loss(logits,
                           target_batch[[2]],
                           weight=FLAGS$logit_weight)
        }else if(FLAGS$supervised == 'both'){
            logits = encoderModel_embedding_to_logit(tf$concat(list(source_emb, target_emb), axis=as.integer(0)), num_labels)
            encoderModel_add_logit_loss(logits,
                           tf$concat(list(source_batch[[2]], target_batch[[2]]), axis=as.integer(0)),
                           weight=FLAGS$logit_weight)
        }else{print("Invalid FLAGS$supervised"); quit("no");}
    }

    ## Global training step
    step = tf$train$get_or_create_global_step()

    ## Use a placeholder in the graph for user-defined learning rate
    decay_step = tf$placeholder(tf$float32)

    ## Set up learning rate
    t_learning_rate = .learning_rate(step, decay_step, FLAGS)

    ## Create training operation
    train_op = encoderModel_create_train_op(t_learning_rate, step)
    loss_op = tf$losses$get_total_loss()
    summary_op = tf$summary$merge_all()

    if(FLAGS$log.results == TRUE){

      ## Write summaries
      summary_writer = tf$summary$FileWriter(file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), "/")), graph)

      ## Save model
      saver <- tf$train$Saver(max_to_keep=FLAGS$max_checkpoints,
                              keep_checkpoint_every_n_hours=FLAGS$keep_checkpoint_every_n_hours)
    }
  }) ## End graphdef

  ## Training scope
  sess = NULL ## Appease R check
  loss_tracker = c(); patience_count = 0;
  with(tf$Session(graph=graph, config=config) %as% sess, {
      ## Set the logging level for tensorflow to only fatal issues
      tf$logging$set_verbosity(tf$logging$FATAL)

      ## Define seed at the graph-level
      ## From docs: If the graph-level seed is set, but the operation seed is not:
      ## The system deterministically picks an operation seed in conjunction with
      ## the graph-level seed so that it gets a unique random sequence.
      if(FLAGS$random_seed != 0){tf$set_random_seed(FLAGS$random_seed)}

      ## Normalize full data matrix
      if(FLAGS$full_norm == TRUE){
        source_data = sess$run(tf$nn$l2_normalize(source_data, axis=as.integer(1)))
        target_data = sess$run(tf$nn$l2_normalize(target_data, axis=as.integer(1)))
      }

      ## Initialize everything
      tf$global_variables_initializer()$run()
      print("Done random initialization")

      ## Create results dir
      if(FLAGS$log.results == TRUE){
        dir.create(file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), '/train/')), showWarnings = FALSE)
      }

      # ## Assert that nothing more can be added to the graph
      # #tf$get_default_graph().finalize()

      ## Initialize the Dataset iterators and feed correct arguments based on supervised vs. unsupervised
      if(FLAGS$supervised == "none"){
        sess$run(source_iter$initializer, feed_dict=dict(source_data_ph = source_data))
      }else if(is.element(FLAGS$supervised, c("source", "both"))){
        sess$run(source_iter$initializer, feed_dict=dict(source_data_ph = source_data, source_labels_ph = source_labels))
      }

      if(FLAGS$supervised == "none"){
        sess$run(target_iter$initializer, feed_dict=dict(target_data_ph = target_data))
      }else if(is.element(FLAGS$supervised, c("target", "both"))){
        sess$run(target_iter$initializer, feed_dict=dict(target_data_ph = target_data, target_labels_ph = target_labels))
      }

      ## Training!
      for(step in seq_len(as.integer(FLAGS$max_steps))){
        if(patience_count >= 0){
          res = sess$run(list(train_op, summary_op, loss_op), feed_dict=dict(decay_step = FLAGS$decay_step))
        }else{
          res = sess$run(list(train_op, summary_op, loss_op), feed_dict=dict(decay_step = 100))
        }

        if(((step %% 100) == 0) | (step == 1)){ print(paste0("Step: ", step, "    Loss: ", round(res[[3]], 4))) }

        ## Record loss
        loss_tracker[step] = as.numeric(res[[3]])
        if(FLAGS$early_stopping == TRUE){
          patience_count = .early_stop(loss_tracker, step, patience_count, early_stopping_active=floor(FLAGS$max_steps/5), min_delta=0.01)
        }

        ## Exceeded patience, now ready to stop. (Patience is == TRUE once patience_count >= 50, otherwise [0-49])
        if(is.logical(patience_count)){
          FLAGS$max_steps = step + 1000 ## 1000 more steps with fast learning_rate decay
          FLAGS$early_stopping = FALSE
          patience_count = -1
          print("=========================================================")
          print("================ EARLY STOPPING TRIGGERED ===============")
          print("==FINALIZING OPTIMIZATION WITH FAST LEARNING RATE DECAY==")
          print("=========================================================")
          print(paste0("Step: ", step, "    Loss: ", round(res[[3]], 4)))
        }

        ## Save embeddings and walker probabilities + summaries
        if(((step %% FLAGS$log_every_n_steps == 0) | (step == 1) | (step == FLAGS$max_steps)) & FLAGS$log.results == TRUE){
          if(step > 1){ print("==================== Saving results =====================") } ## Suppress first save
          ## Save embeddings
          emb = encoderModel_calc_embedding(sess, rbind(source_data, target_data), test_emb, test_in, FLAGS)
          write.table(emb, file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), '/train/emb_activations_', as.character(step), '.csv')), sep=",", col.names=FALSE, row.names=FALSE)

          ## Plot at every loggin step
          if((FLAGS$plot==TRUE) & (step > 1)){
            .plotTSNE(emb, c(rep(obj1_name, nrow(source_data)), rep(obj2_name, nrow(target_data))), file_out=paste0(file.path(FLAGS$logdir, paste0('model_full/plots/step_', as.character(step), '.png'))))
          }
          ## Write summaries
          summary_writer$add_summary(res[[2]], step)

          ## Write out graph
          saver$save(sess, file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), '/train', '/model.ckpt')), as.integer(step))
        }
        ## Complete training
        if(step == FLAGS$max_steps){
          if(FLAGS$log.results == FALSE){ emb = encoderModel_calc_embedding(sess, rbind(source_data, target_data), test_emb, test_in, FLAGS) }
          return(emb)
        }
      }
  })
  return(NULL)
}
