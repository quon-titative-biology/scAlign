context("scalign")

test_that("scAlign halts on no input", {

  ## Run scAlign with high_var_genes
  expect_error(scAlign(encoder.data="scale.data",
                       supervised='none',
                       run.encoder=TRUE,
                       run.decoder=TRUE,
                       log.dir=file.path(results.dir, 'models','gene_input'),
                       device="CPU"))
})

test_that("scAlign object creator halts on none SCE input", {

  ## Input data, 1000 genes x 100 cells
  data = matrix(sample.int(10000, 1000*100, TRUE), 1000, 100)
  rownames(data) = paste0("gene", seq_len(1000))
  colnames(data) = paste0("cell", seq_len(100))

  age    = c(rep("young",50), rep("old",50))
  labels = c(c(rep("type1",25), rep("type2",25)), c(rep("type1",25), rep("type2",25)))

  ctrl.data = data[,which(age == "young")]
  stim.data = data[,which(age == "old")]

  ## Build the scAlign class object and compute PCs
  expect_error(scAlignCreateObject(sce.objects = list("YOUNG"=ctrl.data,
                                                      "OLD"=stim.data),
                                   labels = list(cell_type[which(cell_age == "young")],
                                                 cell_type[which(cell_age == "old")]),
                                   pca.reduce = TRUE,
                                   pcs.compute = 50,
                                   cca.reduce = TRUE,
                                   ccs.compute = 15,
                                   project.name = "scAlign_Kowalcyzk_HSC"))
})


test_that("Alignment produces consistent results", {

  library(scAlign)
  library(SingleCellExperiment)
  library(class)
  library(ggplot2)

  ## Load in cellbench data
  data("cellbench", package = "scAlign", envir = environment())

  ## Extract RNA mixture cell types
  mix.types = unlist(lapply(strsplit(colnames(cellbench), "-"), "[[", 2))
  ## Extract Platform
  batch = c(rep("CEL", length(which(!grepl("sortseq", colnames(cellbench)) == TRUE))),
            rep("SORT", length(which(grepl("sortseq", colnames(cellbench)) == TRUE))))

  ## Create SCE objects to pass into scAlignCreateObject
  youngMouseSCE <- SingleCellExperiment(
      assays = list(scale.data = cellbench[,batch=='CEL'])
  )

  oldMouseSCE <- SingleCellExperiment(
      assays = list(scale.data = cellbench[,batch=='SORT'])
  )

  ## Build the scAlign class object and compute PCs
  scAlignCB = scAlignCreateObject(sce.objects = list("CEL"=youngMouseSCE,
                                                     "SORT"=oldMouseSCE),
                                   labels = list(mix.types[batch=='CEL'],
                                                 mix.types[batch=='SORT']),
                                   data.use="scale.data",
                                   pca.reduce = FALSE,
                                   cca.reduce = TRUE,
                                   ccs.compute = 5,
                                   project.name = "scAlign_cellbench")

   ## Run scAlign with all_genes
   scAlignCB = scAlign(scAlignCB,
                       options=scAlignOptions(steps=500,
                                              log.every=500,
                                              norm=TRUE,
                                              early.stop=FALSE),
                       encoder.data="scale.data",
                       supervised='none',
                       run.encoder=TRUE,
                       run.decoder=FALSE,
                       log.dir=file.path('~/models_temp','gene_input'),
                       device="CPU")

  aligned_data  = reducedDim(scAlignCB, "ALIGNED-GENE")
  aligned_CEL = aligned_data[which(batch == "CEL"),]
  aligned_SORT   = aligned_data[which(batch == "SORT"),]

  class_res = knn(aligned_CEL, aligned_SORT, mix.types[which(batch == "CEL")], k=15)
  class_acc = mean(class_res == mix.types[which(batch == "SORT")])
  expect_gte(class_acc, 0.5)

})
