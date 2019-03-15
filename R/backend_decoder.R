#' Compute projection for mini-batch
#'
#' Takes the embeddings and returns high dimensional data
#'
#' @return Tensorflow op
#'
#' @param model_func Network op in graph
#' @param data Data to pass through model_func
#' @param is_training Determines whether dropout and batch norm are run.
#'
#' @import tensorflow
#'
#' @keywords internal
decoderModel_emb_to_proj = function(model_func, data, is_training=TRUE){
  with(tf$variable_scope('net_decode', reuse=is_training), {
    return(model_func(data, is_training=is_training))
  })
}

#' Compute projections for complete data after training
#'
#' Takes the embeddings and returns high dimensional data
#'
#' @return Tensorflow op
#'
#' @param sess Current tensorflow session
#' @param data Input matrix or tensor to reduce to embedding space
#' @param endpoint Operation to evaluate in the graph
#' @param test_in Placeholder to process full data in batches
#' @param batch_size how many cells per each batch
#' @param FLAGS Tensorflow run arguments
#'
#' @import tensorflow
#'
#' @keywords internal
decoderModel_calc_projection = function(sess, data, data_full, endpoint, test_in, FLAGS, batch_size=150){
  proj = matrix(nrow=nrow(data), ncol=ncol(data_full), 0.0)
  for(i in seq(1,nrow(data),batch_size)){
    ix_end = min((i+(batch_size-1)), nrow(data))
    proj[i:ix_end,] = sess$run(endpoint, dict(test_in=data[i:ix_end,]))
  }
  return(proj)
}

#' Convert embedding to logit scores
#'
#' @return Tensorflow op
#'
#' @param proj Current project of data during training
#' @param data High dimensional data
#' @param mode Dataste being operated on
#' @param debug Verbosity of command line output
#'
#' @import tensorflow
#'
#' @keywords internal
decoderModel_add_mse_loss = function(proj, data, mode, debug=TRUE){
  ## Define decoder reconstruction loss
  loss_decoder = tf$losses$mean_squared_error(data,
                                              proj,
                                              weights=1.0,
                                              scope=NULL,
                                              loss_collection=tf$GraphKeys$LOSSES,
                                              reduction=tf$losses$Reduction$SUM_BY_NONZERO_WEIGHTS)

  ## Record loss
  tf$summary$scalar(paste0('Loss_decoder_', mode), loss_decoder)
}

#' Builds training operation
#'
#' Defines how the model is optimized and update scheme.
#'
#' @return Tensorflow training op
#'
#' @param learning_rate Learning rate for optimizer
#' @param step Global training step
#'
#' @import tensorflow
#'
#' @keywords internal
decoderModel_create_train_op = function(learning_rate, step){
  ## Collect all loss components
  train_loss    = tf$losses$get_total_loss()
  ## Minimize loss function
  optimizer = tf$train$AdamOptimizer(learning_rate)
  train_op = optimizer$minimize(train_loss, step)
  ## Monitor
  tf$summary$scalar('Learning_Rate', learning_rate)
  tf$summary$scalar('Loss_Total', train_loss)
  return(train_op)
}
