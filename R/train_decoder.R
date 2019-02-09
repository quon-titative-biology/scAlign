#' decoder for scAlign
#'
#' Projects cells from joint cell state space into one condition.
#'
#' @return final interpolation results
#'
#' @param FLAGS Tensorflow run arguments.
#' @param CV Model that is being run.
#' @param config Hardware configuration options.
#' @param data_full High dimensional data used for reconstruction error
#' @param data_emb Low dimensional data from scAlign encoder post alignment
#' @param all_data_emb Low dimensional cells from all conditions to project
#'
#' @import tensorflow
#' @import tfdatasets
#' @import purrr
#'
#' @keywords internal
decoderModel_train_decoder = function(FLAGS, config, mode,
                                      data_full, data_emb, all_data_emb){

  ## Define network structure
  graph = tf$Graph()
  with(graph$as_default(), {
    print("Graph construction")
    data_emb_ph  = tf$placeholder(tf$float32, shape(NULL,as.integer(ncol(data_emb))))
    data_full_ph = tf$placeholder(tf$float32, shape(NULL,as.integer(ncol(data_full))))
    dataset      = tfdatasets::tensor_slices_dataset(tuple(data_emb_ph, data_full_ph))  %>%
                     tfdatasets::dataset_shuffle(nrow(data_full)) %>%
                     tfdatasets::dataset_repeat() %>%
                     tfdatasets::dataset_batch(FLAGS$unsup_batch_size)
    dataset_iter = tfdatasets::make_iterator_initializable(dataset)
    mini_batch   = tfdatasets::iterator_get_next(dataset_iter)

    model_function_decoder = partial(
        match.fun(paste0("decoder_", FLAGS$decoder)),
        emb_size=FLAGS$emb_size,
        final_dim=as.integer(ncol(data_full)),
        complexity=FLAGS$complexity,
        batch_norm=FLAGS$batch_norm)

    ## Define test first, also acts as network initializer.
    ## tf.get_variable() is only called when reuse=FALSE for named/var scopes...
    test_in = tf$placeholder(tf$float32, shape(NULL, as.integer(ncol(data_emb))), 'test_in')
    test_proj = decoderModel_emb_to_proj(model_function_decoder, test_in, is_training=FALSE)

    ## Global training step
    step = tf$train$get_or_create_global_step()

    ## Define decoder op for training
    proj = decoderModel_emb_to_proj(model_function_decoder, mini_batch[[1]], is_training=TRUE)

    ## Loss
    decoderModel_add_mse_loss(proj, mini_batch[[2]], mode)

    ## Use a placeholder in the graph for user-defined learning rate
    decay_step = tf$placeholder(tf$float32)

    ## Set up learning rate
    t_learning_rate = .learning_rate(step, decay_step, FLAGS)

    ## Create training operation
    train_op = decoderModel_create_train_op(t_learning_rate, step)
    loss_op = tf$losses$get_total_loss()
    summary_op = tf$summary$merge_all()

    if(FLAGS$log.results == TRUE){

      ## Write summaries
      summary_writer = tf$summary$FileWriter(file.path(paste0(FLAGS$logdir, '/', as.character(mode), "_decoder/")), graph)

      ## Save model
      saver <- tf$train$Saver(max_to_keep=FLAGS$max_checkpoints,
                              keep_checkpoint_every_n_hours=FLAGS$keep_checkpoint_every_n_hours)
    }
  }) ## End graphdef

  ## Training scope
  sess = NULL ## Appease R check
  loss_tracker = c(); patience_count = 0; FLAGS$early_stopping = FALSE;
  with(tf$Session(graph=graph, config=config) %as% sess, {
      ## Set the logging level for tensorflow to only fatal issues
      tf$logging$set_verbosity(tf$logging$FATAL)
      ## Define seed at the graph-level
      ## From docs: If the graph-level seed is set, but the operation seed is not:
      ## The system deterministically picks an operation seed in conjunction with
      ## the graph-level seed so that it gets a unique random sequence.s
      if(FLAGS$random_seed != 0){tf$set_random_seed(FLAGS$random_seed)}

      ## Initialize everything
      tf$global_variables_initializer()$run()
      print("Done random initialization")

      ## Initialize dataset iterator
      sess$run(dataset_iter$initializer, feed_dict=dict(data_emb_ph = data_emb, data_full_ph = data_full))

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
          patience_count = .early_stop(loss_tracker, step, patience_count, early_stopping_active=floor(FLAGS$max_steps/10), min_delta=0.01)
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

        ## Save
        if(((step %% FLAGS$log_every_n_steps == 0) | (step == 1) | (step == FLAGS$max_steps)) & FLAGS$log.results == TRUE){
          ## Save projections
          proj = decoderModel_calc_projection(sess, all_data_emb, data_full, test_proj, test_in, FLAGS)
          write.table(proj, file.path(paste0(FLAGS$logdir,'/', as.character(mode), '_decoder', '/projected_data_', as.character(step), '.csv')), sep=",", col.names=FALSE, row.names=FALSE)
          ## Summary reports (Tensorboard)
          summary_writer$add_summary(res[[2]], (step))
          ## Write out graph
          saver$save(sess, file.path(paste0(FLAGS$logdir,'/', as.character(mode), '_decoder', '/model.ckpt')), as.integer(step))
        }
        ## Complete training
        if(step == FLAGS$max_steps){
          if(FLAGS$log.results == FALSE){ proj = decoderModel_calc_projection(sess, all_data_emb, data_full, test_proj, test_in, FLAGS) }
          return(proj)
        }
      }
  })
  return(NULL)
}
