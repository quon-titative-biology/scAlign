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
  library(Seurat)
  library(SingleCellExperiment)
  library(class)

  working.dir = "/home/ucdnjj/lab-data/hsc-age" #where our data file, kowalcyzk_gene_counts.rda is located
  results.dir = "/home/ucdnjj/lab-results"      #where the output should be stored

  ## Load in data
  load(file.path(working.dir, 'kowalcyzk_gene_counts.rda'))

  ## Extract age and cell type labels
  cell_age = unlist(lapply(strsplit(colnames(C57BL6_mouse_data), "_"), "[[", 1))
  cell_type = gsub('HSC', '', unlist(lapply(strsplit(colnames(C57BL6_mouse_data), "_"), "[[", 2)))

  ## Separate young and old data
  young_data = C57BL6_mouse_data[unique(row.names(C57BL6_mouse_data)),which(cell_age == "young")]
  old_data   = C57BL6_mouse_data[unique(row.names(C57BL6_mouse_data)),which(cell_age == "old")]

  ## Set up young mouse Seurat object
  youngMouseSeuratObj <- CreateSeuratObject(raw.data = young_data, project = "MOUSE_AGE", min.cells = 0)
  youngMouseSeuratObj <- FilterCells(youngMouseSeuratObj, subset.names = "nGene", low.thresholds = 100, high.thresholds = Inf)
  youngMouseSeuratObj <- NormalizeData(youngMouseSeuratObj)
  youngMouseSeuratObj <- ScaleData(youngMouseSeuratObj, do.scale=T, do.center=T, display.progress = T)

  ## Set up old mouse Seurat object
  oldMouseSeuratObj <- CreateSeuratObject(raw.data = old_data, project = "MOUSE_AGE", min.cells = 0)
  oldMouseSeuratObj <- FilterCells(oldMouseSeuratObj, subset.names = "nGene", low.thresholds = 100, high.thresholds = Inf)
  oldMouseSeuratObj <- NormalizeData(oldMouseSeuratObj)
  oldMouseSeuratObj <- ScaleData(oldMouseSeuratObj, do.scale=T, do.center=T, display.progress = T)

  ## Create SCE objects to pass into scAlignCreateObject
  youngMouseSCE <- SingleCellExperiment(
      assays = list(counts = t(FetchData(youngMouseSeuratObj, vars.all=rownames(young_data), use.raw=TRUE)),
                    scale.data = t(FetchData(youngMouseSeuratObj, vars.all=rownames(young_data), use.scaled=TRUE)))
  )

  oldMouseSCE <- SingleCellExperiment(
      assays = list(counts = t(FetchData(oldMouseSeuratObj, vars.all=rownames(old_data), use.raw=TRUE)),
                    scale.data = t(FetchData(oldMouseSeuratObj, vars.all=rownames(old_data), use.scaled=TRUE)))
  )

  ## Build the scAlign class object and compute PCs
  scAlignHSC = scAlignCreateObject(sce.objects = list("YOUNG"=youngMouseSCE, "OLD"=oldMouseSCE),
                                   labels = list(cell_type[which(cell_age == "young")], cell_type[which(cell_age == "old")]),
                                   data.use="scale.data",
                                   pca.reduce = TRUE,
                                   pcs.compute = 50,
                                   cca.reduce = TRUE,
                                   ccs.compute = 15,
                                   project.name = "scAlign_Kowalcyzk_HSC")

 
  ## View SCE object
  print(scAlignHSC)

  ## Run scAlign with high_var_genes
  scAlignHSC = scAlign(scAlignHSC,
                       options=scAlignOptions(steps=5000, log.every=1000, norm=TRUE, early.stop=TRUE),
                       encoder.data="scale.data",
                       supervised='none',
                       run.encoder=TRUE,
                       run.decoder=TRUE,
                       log.dir=file.path(results.dir, 'models','gene_input'),
                       device="CPU")

  aligned_data  = reducedDim(scAlignHSC, "ALIGNED-GENE")
  aligned_young = aligned_data[which(cell_age == "young"),]
  aligned_old   = aligned_data[which(cell_age == "old"),]

  class_res = knn(aligned_young, aligned_old, cell_type[which(cell_age == "young")], k=15)
  class_acc = mean(class_res == cell_type[which(cell_age == "old")])
  expect_gte(class_acc, 0.8)
  
})
