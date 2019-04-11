#' Network small
#'
#' Defines network architecture for scAlign.
#'
#' @return Neural network graph op
#'
#' @param inputs Mini-batch placeholder
#' @param input_size Number of features per cell
#' @param complexity Determines the depth and width of an automatically created network
#' @param emb_size Number of hidden nodes in final (embedding) hidden layer
#' @param l2_weight Weight on l2_regularizer
#' @param dropout_rate Probability for dropout.
#' @param is_training Determines if dropout and batch norm should be include in pass through network
#' @param batch_norm Determines if batch normalization layers should be included
#'
#' @import tensorflow
#'
#' @keywords internal
encoder_small = function(inputs,
                         input_size=NULL,
                         complexity=3,
                         emb_size=32,
                         l2_weight=1e-4,
                         dropout_rate=0.3,
                         is_training=TRUE,
                         batch_norm=TRUE){

    inputs = tf$cast(inputs, tf$float32)
    net = tf$layers$dense(inputs=inputs,
                          units=128,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc1')
    if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_1')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop1')
    emb = tf$layers$dense(inputs=net,
                          units=emb_size,
                          activation=NULL,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc2')
    return(emb)
}

#' Network medium
#'
#' Defines network architecture for scAlign.
#'
#' @return Neural network graph op
#'
#' @param inputs Mini-batch placeholder
#' @param input_size Number of features per cell
#' @param complexity Determines the depth and width of an automatically created network
#' @param emb_size Number of hidden nodes in final (embedding) hidden layer
#' @param l2_weight Weight on l2_regularizer
#' @param dropout_rate Probability for dropout.
#' @param is_training Determines if dropout and batch norm should be include in pass through network
#' @param batch_norm Determines if batch normalization layers should be included
#'
#' @import tensorflow
#'
#' @keywords internal
encoder_medium = function(inputs,
                          input_size=NULL,
                          complexity=3,
                          emb_size=32,
                          l2_weight=1e-4,
                          dropout_rate=0.3,
                          is_training=TRUE,
                          batch_norm=TRUE){

    inputs = tf$cast(inputs, tf$float32)
    net = tf$layers$dense(inputs=inputs,
                          units=256,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc1')
    if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_1')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop1')
    net = tf$layers$dense(inputs=net,
                          units=128,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc2')
    if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_2')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop2')
    emb = tf$layers$dense(inputs=net,
                          units=emb_size,
                          activation=NULL,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc3')
    return(emb)
}

#' Network large
#'
#' Defines network architecture for scAlign.
#'
#' @param inputs Mini-batch placeholder
#' @param input_size Number of features per cell
#' @param complexity Determines the depth and width of an automatically created network
#' @param emb_size Number of hidden nodes in final (embedding) hidden layer
#' @param l2_weight Weight on l2_regularizer
#' @param dropout_rate Probability for dropout.
#' @param is_training Determines if dropout and batch norm should be include in pass through network
#' @param batch_norm Determines if batch normalization layers should be included
#'
#' @return Neural network graph op
#'
#' @import tensorflow
#'
#' @keywords internal
encoder_large = function(inputs,
                         input_size=NULL,
                         complexity=3,
                         emb_size=32,
                         l2_weight=1e-4,
                         dropout_rate=0.3,
                         is_training=TRUE,
                         batch_norm=TRUE){

    inputs = tf$cast(inputs, tf$float32)
    net = tf$layers$dense(inputs=inputs,
                          units=512,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc1')
    if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_1')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop1')
    net = tf$layers$dense(inputs=net,
                          units=256,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc2')
    if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_2')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop2')
    net = tf$layers$dense(inputs=net,
                          units=128,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc3')
    if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_3')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop3')
    emb = tf$layers$dense(inputs=net,
                          units=emb_size,
                          activation=NULL,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          name='fc4')
    return(emb)
}

#' Network small
#'
#' Defines decoder network architecture for scAlign.
#'
#' @return Neural network graph op
#'
#' @param inputs Mini-batch placeholder
#' @param complexity Determines the depth and width of an automatically created network
#' @param final_dim Number of features in high dimensional data
#' @param emb_size Number of hidden nodes in final (embedding) hidden layer
#' @param l2_weight Weight on l2_regularizer
#' @param dropout_rate Probability for dropout.
#' @param is_training Determines if dropout and batch norm should be include in pass through network
#' @param batch_norm Determines if batch normalization layers should be included
#'
#' @import tensorflow
#'
#' @keywords internal
decoder_small = function(inputs,
                         complexity,
                         final_dim,
                         emb_size=32,
                         l2_weight=1e-4,
                         dropout_rate=0.3,
                         is_training=TRUE,
                         batch_norm=FALSE){

    inputs = tf$cast(inputs, tf$float32)
    net = tf$layers$dense(inputs=inputs,
                          units=128,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          bias_initializer=tf$ones_initializer(),
                          name='fc1')
    #if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_1')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop1')
    proj = tf$layers$dense(inputs=net,
                           units=final_dim,
                           activation=NULL,
                           kernel_initializer=tf$contrib$layers$xavier_initializer(),
                           kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                           use_bias=TRUE,
                           bias_initializer=tf$ones_initializer(),
                           name='fc2')
    return(proj)
}

#' Network medium
#'
#' Defines network architecture for scAlign.
#'
#' @param inputs Mini-batch placeholder
#' @param complexity Determines the depth and width of an automatically created network
#' @param final_dim Number of features in high dimensional data
#' @param emb_size Number of hidden nodes in final (embedding) hidden layer
#' @param l2_weight Weight on l2_regularizer
#' @param dropout_rate Probability for dropout.
#' @param is_training Determines if dropout and batch norm should be include in pass through network
#' @param batch_norm Determines if batch normalization layers should be included
#'
#' @return Neural network graph op
#'
#' @import tensorflow
#'
#' @keywords internal
decoder_medium = function(inputs,
                          complexity,
                          final_dim,
                          emb_size=32,
                          l2_weight=1e-4,
                          dropout_rate=0.3,
                          is_training=TRUE,
                          batch_norm=FALSE){

    inputs = tf$cast(inputs, tf$float32)
    net = tf$layers$dense(inputs=inputs,
                          units=128,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          bias_initializer=tf$ones_initializer(),
                          name='fc1')
    #if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_1')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop1')
    net = tf$layers$dense(inputs=net,
                          units=256,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          bias_initializer=tf$ones_initializer(),
                          name='fc2')
    #if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_2')}
    net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop2')
    proj = tf$layers$dense(inputs=net,
                           units=final_dim,
                           activation=NULL,
                           kernel_initializer=tf$contrib$layers$xavier_initializer(),
                           kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                           use_bias=TRUE,
                           name='fc3')
    return(proj)
}

#' Network large
#'
#' Defines network architecture for scAlign.
#'
#' @param inputs Mini-batch placeholder
#' @param complexity Determines the depth and width of an automatically created network
#' @param final_dim Number of features in high dimensional data
#' @param emb_size Number of hidden nodes in final (embedding) hidden layer
#' @param l2_weight Weight on l2_regularizer
#' @param dropout_rate Probability for dropout.
#' @param is_training Determines if dropout and batch norm should be include in pass through network
#' @param batch_norm Determines if batch normalization layers should be included
#'
#' @return Neural network graph op
#'
#' @import tensorflow
#'
#' @keywords internal
decoder_large = function(inputs,
                          complexity,
                          final_dim,
                          emb_size=32,
                          l2_weight=1e-4,
                          dropout_rate=0.3,
                          is_training=TRUE,
                          batch_norm=FALSE){

    inputs = tf$cast(inputs, tf$float32)
    net = tf$layers$dense(inputs=inputs,
                          units=128,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          bias_initializer=tf$ones_initializer(),
                          name='fc1')
    #if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_1')}
    #net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop1')
    net = tf$layers$dense(inputs=net,
                          units=256,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          bias_initializer=tf$ones_initializer(),
                          name='fc2')
    #if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_2')}
    #net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop2')
    net = tf$layers$dense(inputs=net,
                          units=512,
                          activation=tf$nn$relu,
                          kernel_initializer=tf$contrib$layers$xavier_initializer(),
                          kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                          use_bias=TRUE,
                          bias_initializer=tf$ones_initializer(),
                          name='fc3')
    #if(batch_norm == TRUE){net = tf$layers$batch_normalization(net, training=is_training, name='batch_norm_3')}
    #net = tf$layers$dropout(inputs=net, rate=dropout_rate, training=is_training, name='drop3')
    proj = tf$layers$dense(inputs=net,
                           units=final_dim,
                           activation=NULL,
                           kernel_initializer=tf$contrib$layers$xavier_initializer(),
                           kernel_regularizer=tf$contrib$layers$l2_regularizer(l2_weight),
                           use_bias=TRUE,
                           bias_initializer=tf$ones_initializer(),
                           name='fc4')
    return(proj)
}
