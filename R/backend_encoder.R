#' Add walker loss component
#'
#' Creates the walker loss component using embedding representation of
#' high-dimensional data. Specifies random walk procedure.
#'
#' @return Tensorflow loss op
#'
#' @param p_target Pairwise cell-cell similarity
#' @param a Data embeddings
#' @param b Data embeddings
#' @param data_shape Dimension of data
#' @param walker_weight Walker loss component weight
#' @param visit_weight Visit loss component weight (unused)
#' @param mode Specifies which walker loss is being computed
#' @param comp How to compute round trip probabilities
#' @param betas Value for fixed beta gaussian kernel
#' @param diag Whether to include self similarity
#' @param debug Sets the verbosity level
#'
#' @import tensorflow
#'
#' @keywords internal
encoderModel_add_semisup_loss_data_driven = function(p_target, a, b, data_shape, walker_weight=1.0, visit_weight=1.0, mode='source', comp='exp', betas=NULL, diag="zero", debug=TRUE){
  with(tf$compat$v1$name_scope(paste0('loss_walker_', mode)), {
    ## Distance in embedding space
    match_ab = tf$compat$v1$matmul(a, b, transpose_b=TRUE, name=paste0('match_ab_', mode))
    p_ab = tf$compat$v1$nn$softmax(match_ab, name=paste0('p_ab_', mode))
    p_ba = tf$compat$v1$nn$softmax(tf$compat$v1$transpose(match_ab), name=paste0('p_ba_', mode))
    p_aba = tf$compat$v1$matmul(p_ab, p_ba, name=paste0('p_aba_', mode))

    ## Remove diagonal, interested in pairwise similarities not self.
    if(diag == "zero"){
      p_aba = tf$compat$v1$matrix_set_diag(p_aba, tf$compat$v1$zeros(shape(data_shape)))
    }

    ## Conditional probabilities.
    loss_aba = tf$compat$v1$losses$softmax_cross_entropy(
        p_target,
        tf$compat$v1$log(1e-20 + p_aba),
        weights=walker_weight,
        scope=paste0('loss_aba_', mode))

    ## Histograms for tensorboard
    tf$compat$v1$summary$histogram(paste0('p_ab_', mode), p_ab)
    tf$compat$v1$summary$histogram(paste0('p_ba_', mode), p_ba)
    tf$compat$v1$summary$histogram(paste0('p_aba_', mode), p_aba)
    tf$compat$v1$summary$histogram('emb_a', a)
    tf$compat$v1$summary$histogram('emb_b', b)
    tf$compat$v1$summary$histogram('T_ij', p_target)
    tf$compat$v1$summary$histogram('sum_p_ab', tf$compat$v1$reduce_sum(p_ab, as.integer(-1)))
    tf$compat$v1$summary$histogram('sum_p_ba', tf$compat$v1$reduce_sum(p_ba, as.integer(-1)))
    tf$compat$v1$summary$scalar('dot_min', tf$compat$v1$reduce_min(match_ab))
    tf$compat$v1$summary$scalar('dot_max', tf$compat$v1$reduce_max(match_ab))
    tf$compat$v1$summary$scalar('nnzero-emb_a', tf$compat$v1$count_nonzero(tf$compat$v1$cast(a, tf$compat$v1$float32)))
    tf$compat$v1$summary$scalar('nnzero-emb_b', tf$compat$v1$count_nonzero(tf$compat$v1$cast(b, tf$compat$v1$float32)))

    ## Record loss
    tf$compat$v1$summary$scalar(paste0('Loss_aba_', mode), loss_aba)
  })
}

#' Add classifier loss component
#'
#' Cross entropy of labels and predictions
#'
#' @return Tensorflow loss op
#'
#' @param logits cell x label logits
#' @param labels cell x 1 true labels
#' @param weight Scale of classifier loss component
#' @param smoothing tensorflow label_smoothing (unused)
#'
#' @import tensorflow
#'
#' @keywords internal
encoderModel_add_logit_loss = function(logits, labels, weight=1.0){
   weights = tf$compat$v1$clip_by_value(tf$compat$v1$math$add(labels, tf$compat$v1$constant(as.integer(1), dtype=tf$compat$v1$int32)), as.integer(0), as.integer(1))
   weight = tf$compat$v1$math$multiply(tf$compat$v1$cast(weights,dtype=tf$compat$v1$float32), weight)
   tf$compat$v1$summary$histogram('weights_label', weight)
   ## Add clossifier
   with(tf$compat$v1$name_scope('loss_classifier'), {
       logit_loss = tf$compat$v1$losses$softmax_cross_entropy(
           tf$compat$v1$one_hot(labels, logits$get_shape()[-1]),
           logits,
           scope='loss_logit',
           weights=weight)
       # self.add_average(logit_loss)
       tf$compat$v1$summary$scalar('Loss_Logit', logit_loss)
       tf$compat$v1$summary$histogram('logits', logits)
       logit_loss = tf$compat$v1$Print(logit_loss, list(logit_loss), message="class: ")
   })
}

#' Guassian kernel
#'
#' Tensorflow implementation of tsne's gaussian_kernel, accepts tensors of
#' float64 (for precision) and returns tensors of float32
#'
#' @return Tensorflow op
#'
#' @param data cell x feature data matrix
#' @param data_b second cell x feature data matrix, if none provided then data_b = data
#' @param dim number of cells in of mini-batch
#' @param perplexity neighborhood parameter for gaussian kernel
#' @param method distance metric prior to computing probabilities
#' @param diag indicator for self similarity
#'
#' @import tensorflow
#'
#' @keywords internal
encoderModel_gaussian_kernel = function(data, data_b=NULL, dim, perplexity=30, method="euclidean", betas.fixed=0, diag="zero"){
  with(tf$compat$v1$name_scope("gaussian_kernel"), {
    if(is.null(data_b)){data_b = data}

    ## Constants
    PERPLEXITY_TOLERANCE = tf$compat$v1$constant(1e-5, dtype=tf$compat$v1$float64)
    EPSILON_DBL = tf$compat$v1$constant(1e-8, dtype=tf$compat$v1$float64)
    INFINITY    = tf$compat$v1$constant(1e10, dtype=tf$compat$v1$float64)
    perplexity = tf$compat$v1$log(tf$compat$v1$constant(perplexity, dtype=tf$compat$v1$float64))

    ## Variables used in binary search
    betas      = tf$compat$v1$ones(shape(dim), dtype=tf$compat$v1$float64, name="betas")
    p_ij       = tf$compat$v1$ones(shape(dim,dim), dtype=tf$compat$v1$float64, name="p_ij")
    p_ij_unnorm = tf$compat$v1$ones(shape(dim,dim), dtype=tf$compat$v1$float64, name="p_ij_unnorm")
    beta_min   = tf$compat$v1$fill(dims=shape(dim), value=tf$compat$v1$negative(INFINITY))
    beta_max   = tf$compat$v1$fill(dims=shape(dim), value=INFINITY)

    ## distances
    if(method == "euclidean"){
      norm_a = tf$compat$v1$reshape(tf$compat$v1$reduce_sum(tf$compat$v1$square(data), as.integer(1)), shape(-1, 1))
      norm_b = tf$compat$v1$reshape(tf$compat$v1$reduce_sum(tf$compat$v1$square(data_b), as.integer(1)), shape(1, -1))
      affinities = norm_a - tf$compat$v1$constant(2, dtype=tf$compat$v1$float64)*tf$compat$v1$matmul(data, data_b, transpose_b=TRUE) + norm_b
    }

    # if(betas.fixed!=0){
    #   betas = tf$compat$v1$fill(dims=shape(dim), value=tf$compat$v1$cast(betas.fixed, tf$compat$v1$float64))
    #   p_ij = tf$compat$v1$exp(tf$compat$v1$multiply(tf$compat$v1$negative(affinities), tf$compat$v1$reshape(betas, shape(-1,1))))
    #   p_ij = tf$compat$v1$reshape(p_ij, shape(dim, dim)) ## tensorflow R requires this?
    #   ## Reset exp(0)'s to 0
    #   if(diag == "zero"){
    #     p_ij = tf$compat$v1$matrix_set_diag(p_ij, tf$compat$v1$cast(tf$compat$v1$zeros(shape(dim)), tf$compat$v1$float64))
    #   }
    #
    #   sum_pij = tf$compat$v1$reduce_sum(p_ij, axis=as.integer(-1))
    #   sum_pij = tf$compat$v1$where(tf$compat$v1$equal(sum_pij, tf$compat$v1$constant(0, dtype=tf$compat$v1$float64)), tf$compat$v1$fill(dims=shape(dim), value=EPSILON_DBL), sum_pij)
    #
    #   p_ij_unnorm = p_ij
    #   ## Normalize
    #   p_ij = tf$compat$v1$divide(p_ij, tf$compat$v1$reshape(sum_pij, shape(-1,1)))
    #   return(p_ij)
    # }else{

      ## While loop conditional
      cond = function(step, affinities, perplexity, betas, beta_min, beta_max, p_ij, p_ij_unnorm){
        return(tf$compat$v1$less_equal(step, tf$compat$v1$constant(100)))
      }

      ## While loop body
      body = function(step, affinities, perplexity, betas, beta_min, beta_max, p_ij, p_ij_unnorm){
        step = tf$compat$v1$add(step, tf$compat$v1$constant(1))
        ## Compute probabilities
        p_ij = tf$compat$v1$exp(tf$compat$v1$multiply(tf$compat$v1$negative(affinities), tf$compat$v1$reshape(betas, shape(-1,1))))
        p_ij = tf$compat$v1$reshape(p_ij, shape(dim, dim)) ## tensorflow R requires this?
        ## Reset exp(0)'s to 0
        if(diag == "zero"){
          p_ij = tf$compat$v1$matrix_set_diag(p_ij, tf$compat$v1$cast(tf$compat$v1$zeros(shape(dim)), tf$compat$v1$float64))
        }

        sum_pij = tf$compat$v1$reduce_sum(p_ij, axis=as.integer(-1))
        sum_pij = tf$compat$v1$where(tf$compat$v1$equal(sum_pij, tf$compat$v1$constant(0, dtype=tf$compat$v1$float64)), tf$compat$v1$fill(dims=shape(dim), value=EPSILON_DBL), sum_pij)

        p_ij_unnorm = p_ij
        ## Normalize
        p_ij = tf$compat$v1$divide(p_ij, tf$compat$v1$reshape(sum_pij, shape(-1,1)))

        ## Precision based method
        sum_disti_pij = tf$compat$v1$reduce_sum(tf$compat$v1$multiply(affinities, p_ij), axis=as.integer(-1))

        ## Compute entropy
        entropy = tf$compat$v1$add(tf$compat$v1$log(sum_pij), tf$compat$v1$multiply(betas, sum_disti_pij))
        entropy_diff = tf$compat$v1$subtract(entropy, perplexity)

        ## Logic for binary search: Lines 98-109 in tsne_util.pyx.
        ## Set beta_min
        beta_min = tf$compat$v1$where(tf$compat$v1$logical_and(tf$compat$v1$greater(entropy_diff, tf$compat$v1$constant(0.0, dtype=tf$compat$v1$float64)),
                                           tf$compat$v1$greater(tf$compat$v1$abs(entropy_diff), PERPLEXITY_TOLERANCE)),
                                           betas, beta_min)

        ## Set betas for which entropy_diff > 0.0
        ## Where = (cond, x, y): returns elements of x for which cond is true and y elements for which cond is False.
        betas = tf$compat$v1$where(tf$compat$v1$logical_and(tf$compat$v1$greater(entropy_diff, tf$compat$v1$constant(0.0, dtype=tf$compat$v1$float64)),
                                        tf$compat$v1$greater(tf$compat$v1$abs(entropy_diff), PERPLEXITY_TOLERANCE)),
                                        tf$compat$v1$where(tf$compat$v1$equal(beta_max, INFINITY), tf$compat$v1$multiply(betas,2.0), tf$compat$v1$divide(tf$compat$v1$add(betas, beta_max), 2.0)), betas)

        ## Set beta_max
        beta_max = tf$compat$v1$where(tf$compat$v1$logical_and(tf$compat$v1$less_equal(entropy_diff, tf$compat$v1$constant(0.0, dtype=tf$compat$v1$float64)),
                                           tf$compat$v1$greater(tf$compat$v1$abs(entropy_diff), PERPLEXITY_TOLERANCE)),
                                           betas, beta_max)

        ##Set betas for which entropy_diff <= 0.0
        betas = tf$compat$v1$where(tf$compat$v1$logical_and(tf$compat$v1$less_equal(entropy_diff, tf$compat$v1$constant(0.0, dtype=tf$compat$v1$float64)),
                                        tf$compat$v1$greater(tf$compat$v1$abs(entropy_diff), PERPLEXITY_TOLERANCE)),
                                        tf$compat$v1$where(tf$compat$v1$equal(beta_min, tf$compat$v1$negative(INFINITY)), tf$compat$v1$divide(betas,2.0), tf$compat$v1$divide(tf$compat$v1$add(betas, beta_min), 2.0)), betas)

        ## Control dependencies before looping can continue
        with(tf$compat$v1$control_dependencies(list(step, affinities, perplexity, betas, beta_min)), {
          return(list(step, affinities, perplexity, betas, beta_min, beta_max, p_ij, p_ij_unnorm))
        })
      }

      ## Run binary search for bandwidth (precision)
      step = tf$compat$v1$constant(0)
      loop = tf$compat$v1$while_loop(cond, body, list(step, affinities, perplexity, betas, beta_min, beta_max, p_ij, p_ij_unnorm), back_prop=FALSE)
      return(loop[[7]])
    # }
  })
}

#' Compute embedding for mini-batch
#'
#' Takes the high dimensional data through the neural network to embedding layer
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
encoderModel_data_to_embedding = function(model_func, data, is_training=TRUE, scope=''){
  ## Create a graph, transforming data into embeddings
  with(tf$compat$v1$name_scope(paste0('data_to_emb_net', scope)), {
    with(tf$compat$v1$variable_scope('net', reuse=is_training), {
      return(model_func(data, is_training=is_training))
    })
  })
}

#' Convert embedding to logit scores
#'
#' @return Tensorflow op
#'
#' @param data Data to pass through model_func.
#' @param num_labels How many unique labels there are.
#' @param is_training Determines whether dropout and batch norm are run.
#'
#' @import tensorflow
#'
#' @keywords internal
encoderModel_embedding_to_logit = function(data, num_labels, is_training=TRUE){
  ## Add to computation graph, transform embeddings into logit scores
  with(tf$compat$v1$name_scope('emb_to_logit_net'), {
    with(tf$compat$v1$variable_scope('net', reuse=is_training), {
      return(tf$compat$v1$layers$dense(inputs=data,
                             units=num_labels,
                             activation=NULL,
                             kernel_regularizer=tf$keras$regularizers$l2(1e-4),
                             use_bias=TRUE,
                             name='logit_fc'))
    })
  })
}

#' Compute embedding for complete data after training
#'
#' Takes the high dimensional data through the neural network to embedding layer
#'
#' @return Aligned data embedding
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
encoderModel_calc_embedding = function(sess, data, endpoint, test_in, FLAGS, batch_size=150){
  emb = matrix(nrow=nrow(data), ncol=FLAGS$emb_size, 0)
  for(i in seq(1,nrow(data),batch_size)){
    ix_end = min((i+(batch_size-1)), nrow(data))
    emb[i:ix_end,] = sess$run(endpoint, dict(test_in=data[i:ix_end,,drop=FALSE]))
  }
  return(emb)
}

#' Compute embedding for complete data after training
#'
#' Takes the high dimensional data through the neural network to embedding layer
#'
#' @return Aligned data embedding
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
encoderModel_calc_logit = function(sess, emb, endpoint, test_logit_in, FLAGS){
  logit =  sess$run(endpoint, dict(test_logit_in=emb))
  return(logit)
}

# ## Compute logit scores
# def classify(self, data):
#   return self.calc_embedding(data, self.test_logit)

#' Builds training operation
#'
#' Defines how the model is optimized and update scheme.
#'
#' @param learning_rate Learning rate for optimizer
#' @param step Global training step
#'
#' @return Tensorflow training op
#'
#' @import tensorflow
#'
#' @keywords internal
encoderModel_create_train_op = function(learning_rate, step){
  ## Collect all loss components
  train_loss    = tf$compat$v1$losses$get_total_loss()
  ## Minimize loss function
  optimizer = tf$compat$v1$train$AdamOptimizer(learning_rate)
  train_op = optimizer$minimize(train_loss, step)
  ## Monitor
  tf$compat$v1$summary$scalar('Learning_Rate', learning_rate)
  tf$compat$v1$summary$scalar('Loss_Total', train_loss)
  return(train_op)
}
