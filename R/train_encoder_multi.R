#' Computes kernel matrix
#'
#' Adds the defined kernel to the operation graph for high dimensional pairwise data similarity
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
#' @import tensorflow
#'
#' @keywords internal
compute_kernel_matrix = function(data, data_shape, labels=NULL, method=NULL, perplexity=30, diag="zero"){
    ## Defines the similarity matrix T used for asssociation loss
    kernel = encoderModel_gaussian_kernel(data=tf$compat$v1$cast(data, tf$compat$v1$float64), dim=data_shape, perplexity=perplexity, diag=diag)
    ## Cast down for consistent data type
    return(tf$compat$v1$cast(kernel, tf$compat$v1$float32))
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
encoderModel_train_encoder_multi = function(FLAGS, CV, config,
                                            num_labels, data_shape,
                                            names,
                                            data,
                                            labels){

  ## Define network structure
  graph = tf$compat$v1$Graph()
  with(graph$as_default(), {
    print("Graph construction")
    # Create function that defines the network.
    model_function = partial(
        match.fun(paste0("encoder_", FLAGS$encoder)),
        input_size=data_shape,
        complexity=3,
        emb_size=FLAGS$emb_size,
        dropout = FLAGS$dropout,
        batch_norm=FLAGS$batch_norm)

    ## Define test first, also acts as network initializer.
    ## tf.get_variable() is only called when reuse=FALSE for named/var scopes...
    test_in = tf$compat$v1$placeholder(tf$compat$v1$float32, shape(NULL,data_shape), 'test_in')
    test_emb = encoderModel_data_to_embedding(model_function, test_in, is_training=FALSE)
    test_logit = encoderModel_embedding_to_logit(test_emb, num_labels, is_training=FALSE)

    ## Set up inputs. Mini-batch via placeholder and dataset iterators
    for(name in unique(names)){
      with(tf$compat$v1$variable_scope(name), {
        data_ph = tf$compat$v1$placeholder(tf$compat$v1$float32, shape(NULL, data_shape))
        dataset = tf$compat$v1$data$Dataset$from_tensor_slices(data_ph)
        dataset = dataset$shuffle(dim(data)[1])
        dataset = dataset$`repeat`()
        dataset = dataset$batch(FLAGS$unsup_batch_size)
        iter    = dataset$make_initializable_iterator()
        batch   = iter$get_next()
        batch   = tf$compat$v1$nn$l2_normalize(batch, axis=as.integer(1), name=paste0("data"))
      })
      emb = encoderModel_data_to_embedding(model_function, batch, is_training=TRUE, scope=paste0("_",name))
      emb = tf$compat$v1$identity(emb, name=paste0(name,"_emb"))
    }

    if(FLAGS$ref %in% unique(names)){datasets=FLAGS$ref}else{datasets=unique(names)}
    for(name_ref in datasets){
      source_data = graph$get_tensor_by_name(paste0(name_ref,"/data:0"))
      source_emb = graph$get_tensor_by_name(paste0(name_ref,"_emb:0"))
      with(tf$compat$v1$name_scope(name_ref), {
        for(name_target in setdiff(unique(names), name_ref)){
          target_emb = graph$get_tensor_by_name(paste0(name_target,"_emb:0"))
          with(tf$compat$v1$variable_scope(name_ref), {
            print(paste0("Adding ", paste0(name_ref,"-",name_target)))
            ## Add source assoc loss
            T_source = compute_kernel_matrix(data=source_data, data_shape=FLAGS$unsup_batch_size, method=FLAGS$kernel, perplexity=FLAGS$perplexity)
            encoderModel_add_semisup_loss_data_driven(T_source,
                                                       source_emb,
                                                       target_emb,
                                                       FLAGS$unsup_batch_size, ## Replace for balanced case
                                                       walker_weight=FLAGS$walker_weight,
                                                       mode='source',
                                                       comp=FLAGS$prob_comp,
                                                       debug=FLAGS$verbose)
          })
        }
      })
    }
    ## Global training step
    step = tf$compat$v1$train$get_or_create_global_step()

    ## Use a placeholder in the graph for user-defined learning rate
    decay_step = tf$compat$v1$placeholder(tf$compat$v1$float32)

    ## Set up learning rate
    t_learning_rate = .learning_rate(step, decay_step, FLAGS)

    ## Create training operation
    train_op = encoderModel_create_train_op(t_learning_rate, step)
    loss_op = tf$compat$v1$losses$get_total_loss()
    summary_op = tf$compat$v1$summary$merge_all()

    if(FLAGS$log.results == TRUE){

      ## Write summaries
      summary_writer = tf$compat$v1$summary$FileWriter(file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), "/")), graph)

      ## Save model
      saver <- tf$compat$v1$train$Saver(max_to_keep=FLAGS$max_checkpoints,
                              keep_checkpoint_every_n_hours=FLAGS$keep_checkpoint_every_n_hours)
    }
  }) ## End graphdef

  ## Training scope
  sess = NULL ## Appease R check
  loss_tracker = c(); patience_count = 0;
  with(tf$compat$v1$Session(graph=graph, config=config) %as% sess, {
      ## Set the logging level for tensorflow to only fatal issues
      tf$compat$v1$logging$set_verbosity(tf$compat$v1$logging$FATAL)

      ## Define seed at the graph-level
      ## From docs: If the graph-level seed is set, but the operation seed is not:
      ## The system deterministically picks an operation seed in conjunction with
      ## the graph-level seed so that it gets a unique random sequence.
      if(FLAGS$random_seed != 0){tf$compat$v1$set_random_seed(FLAGS$random_seed)}

      ## Normalize full data matrix
      # if(FLAGS$full_norm == TRUE){
      #   source_data = sess$run(tf$compat$v1$nn$l2_normalize(source_data, axis=as.integer(1)))
      #   target_data = sess$run(tf$compat$v1$nn$l2_normalize(target_data, axis=as.integer(1)))
      # }

      # for(op in tf$compat$v1$get_default_graph()$get_operations()){print(op$name)}

      ## Initialize everything
      tf$compat$v1$global_variables_initializer()$run()
      print("Done random initialization")

      ## Create results dir
      if(FLAGS$log.results == TRUE){
        dir.create(file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), '/train/')), showWarnings = FALSE)
      }

      # ## Assert that nothing more can be added to the graph
      # #tf$compat$v1$get_default_graph().finalize()

      ## Initialize the Dataset iterators and feed correct arguments based on supervised vs. unsupervised
      for(name in unique(names)){
        iter = tf$compat$v1$get_default_graph()$get_operation_by_name(paste0(name,"/MakeIterator"))
        placeholder = tf$compat$v1$get_default_graph()$get_tensor_by_name(paste0(name,"/Placeholder:0"))
        sess$run(iter, feed_dict=dict(placeholder = data[which(names == name),]))
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
        # if(FLAGS$early_stopping == TRUE){
        #   patience_count = .early_stop(loss_tracker, step, patience_count, early_stopping_active=floor(FLAGS$max_steps/5), min_delta=0.01)
        # }
        #
        # ## Exceeded patience, now ready to stop. (Patience is == TRUE once patience_count >= 50, otherwise [0-49])
        # if(is.logical(patience_count)){
        #   FLAGS$max_steps = step + 1000 ## 1000 more steps with fast learning_rate decay
        #   FLAGS$early_stopping = FALSE
        #   patience_count = -1
        #   print("=========================================================")
        #   print("================ EARLY STOPPING TRIGGERED ===============")
        #   print("==FINALIZING OPTIMIZATION WITH FAST LEARNING RATE DECAY==")
        #   print("=========================================================")
        #   print(paste0("Step: ", step, "    Loss: ", round(res[[3]], 4)))
        # }

        ## Save embeddings and walker probabilities + summaries
        if(((step %% FLAGS$log_every_n_steps == 0) | (step == 1) | (step == FLAGS$max_steps)) & FLAGS$log.results == TRUE){
          if(step > 1){ print("==================== Saving results =====================") } ## Suppress first save
          ## Save embeddings
          data_norm = sess$run(tf$compat$v1$nn$l2_normalize(data, axis=as.integer(1), name=paste0("data")))
          emb = encoderModel_calc_embedding(sess, data_norm, test_emb, test_in, FLAGS)
          write.table(emb, file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), '/train/emb_activations_', as.character(step), '.csv')), sep=",", col.names=FALSE, row.names=FALSE)

          ## Write summaries
          summary_writer$add_summary(res[[2]], step)

          ## Write out graph
          saver$save(sess, file.path(paste0(FLAGS$logdir, '/model_', as.character(CV), '/train', '/model.ckpt')), as.integer(step))
        }
        ## Complete training
        if(step == FLAGS$max_steps){
          if(FLAGS$log.results == FALSE){ emb = encoderModel_calc_embedding(sess, data, test_emb, test_in, FLAGS) }
          sess$close()
          return(list(emb, round(res[[3]], 4)))
        }
      }
  })
  sess$close()
  return(NULL)
}
