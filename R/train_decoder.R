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
#' @import purrr
#'
#' @keywords internal
decoderModel_train_decoder = function(FLAGS, config, mode,
                                      data_full, data_emb, all_data_emb){

  ## Define network structure
  tf$compat$v1$reset_default_graph()
  graph = tf$compat$v1$Graph()
  with(graph$as_default(), {
    print("Graph construction")
    with(tf$compat$v1$variable_scope("decoder", reuse=tf$compat$v1$AUTO_REUSE), {
      with(tf$compat$v1$name_scope("decoder_data"), {
        data_emb_ph  = tf$compat$v1$placeholder(tf$compat$v1$float32, shape(NULL,as.integer(ncol(data_emb))))
        data_full_ph = tf$compat$v1$placeholder(tf$compat$v1$float32, shape(NULL,as.integer(ncol(data_full))))
        dataset      = tf$compat$v1$data$Dataset$from_tensor_slices(tuple(data_emb_ph, data_full_ph))
        dataset      = dataset$shuffle(dim(data_emb)[1])
        dataset      = dataset$`repeat`()
        dataset      = dataset$batch(FLAGS$unsup_batch_size)
        dataset_iter = dataset$make_initializable_iterator()
        mini_batch   = dataset_iter$get_next()
      })

      # model_function_decoder = partial(
      #     match.fun(paste0("decoder_", FLAGS$decoder)),
      #     emb_size=FLAGS$emb_size,
      #     final_dim=as.integer(ncol(data_full)),
      #     complexity=FLAGS$complexity,
      #     dropout=FLAGS$dropout,
      #     batch_norm=FALSE)

      ## Architectures (parameterizing dists)
      decoder_func = tf$compat$v1$make_template('decoder',
                                      match.fun(paste0("decoder_", FLAGS$decoder)),
                                      emb_size=FLAGS$emb_size,
                                      final_dim=as.integer(ncol(data_full)),
                                      dropout=FLAGS$dropout,
                                      l2_weight=1e-10,
                                      batch_norm=FALSE)

      ## Global training step
      global_step = tf$compat$v1$train$get_or_create_global_step()

      ## Define decoder op for training
      proj = decoder_func(inputs=mini_batch[[1]], is_training=TRUE)

      ## Define decoder reconstruction loss
      loss_decoder = tf$compat$v1$losses$mean_squared_error(mini_batch[[2]],
                                                  proj)

      # ## Set up learning rate
      learning_rate = tf$compat$v1$maximum(
                        tf$compat$v1$train$exponential_decay(
                           FLAGS$learning_rate,
                           global_step,
                           5000,
                           FLAGS$decay_factor),
                       FLAGS$minimum_learning_rate)

      ## Create training operation
      train_loss = loss_decoder
      ## Minimize loss function
      with(tf$compat$v1$name_scope("Adam"), {
        optimizer = tf$compat$v1$train$AdamOptimizer(learning_rate)
        train_op = optimizer$minimize(train_loss, global_step=global_step)
      })

      ## Monitor
      tf$compat$v1$summary$scalar('Learning_Rate', learning_rate)
      tf$compat$v1$summary$scalar('Loss_Total', loss_decoder)

      summary_op = tf$compat$v1$summary$merge_all()

      ## Write summaries
      summary_writer = tf$compat$v1$summary$FileWriter(file.path(paste0(FLAGS$logdir, '/', as.character(mode), "_decoder/")), graph)

      ## Save model
      saver <- tf$compat$v1$train$Saver(max_to_keep=FLAGS$max_checkpoints,
                              keep_checkpoint_every_n_hours=FLAGS$keep_checkpoint_every_n_hours)

      ## Define decoder op for training
      test_in = tf$compat$v1$placeholder(tf$compat$v1$float32, shape(NULL, as.integer(ncol(data_emb))), 'test_in')
      test_proj = decoder_func(inputs=test_in, is_training=FALSE)
    })
  }) ## End graphdef


  ## Training scope
  sess = NULL ## Appease R check
  with(tf$compat$v1$Session(graph=graph, config=config) %as% sess, {
      ## Set the logging level for tensorflow to only fatal issues
      tf$compat$v1$logging$set_verbosity(tf$compat$v1$logging$FATAL)
      ## Define seed at the graph-level
      ## From docs: If the graph-level seed is set, but the operation seed is not:
      ## The system deterministically picks an operation seed in conjunction with
      ## the graph-level seed so that it gets a unique random sequence.s
      if(FLAGS$random_seed != 0){tf$compat$v1$set_random_seed(FLAGS$random_seed)}

      ## Initialize everything
      tf$compat$v1$global_variables_initializer()$run()
      print("Done random initialization")

      ## Initialize dataset iterator
      sess$run(dataset_iter$initializer, feed_dict=dict(data_emb_ph = data_emb, data_full_ph = data_full))

      ## Training!
      for(step in seq_len(FLAGS$max_steps_decoder)){

        ## Training!
        res = sess$run(list(train_op, summary_op, loss_decoder, proj, learning_rate, global_step))

        ## Report loss
        if(((step %% 100) == 0) | (step == 1)){ print(paste0("Step: ", step, "    Loss: ", round(res[[3]], 4))); summary_writer$add_summary(res[[2]], (step)); }

        ## Save
        if((step %% FLAGS$log_every_n_steps == 0) & FLAGS$log.results == TRUE) {
          ## Save projections
          # proj_res = sess$run(test_proj, feed_dict=dict(test_in = data_emb))
          proj_res = decoderModel_calc_projection(sess, all_data_emb, data_full, test_proj, test_in, FLAGS)
          write.table(proj_res, file.path(paste0(FLAGS$logdir,'/', as.character(mode), '_decoder', '/projected_data_', as.character(step), '.csv')), sep=",", col.names=FALSE, row.names=FALSE)
          ## Summary reports (Tensorboard)
          summary_writer$add_summary(res[[2]], (step))
          ## Write out graph
          saver$save(sess, file.path(paste0(FLAGS$logdir,'/', as.character(mode), '_decoder', '/model.ckpt')), as.integer(step))
        }
        ## Complete training
        if(step == FLAGS$max_steps_decoder){
          proj_res = decoderModel_calc_projection(sess, all_data_emb, data_full, test_proj, test_in, FLAGS)
          #proj_res = sess$run(test_proj, feed_dict=dict(test_in = all_data_emb))
          return(proj_res)
        }
      }
  })
  return(NULL)
}
