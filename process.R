if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("Rsamtools")
BiocManager::install("GenomicAlignments")
BiocManager::install("profileplyr")
BiocManager::install("rhdf5")
library("Rsamtools")
library("GenomicAlignments")
library("profileplyr")
library("rhdf5")
setwd("D:\\experiment\\deepsignal2")
data1 <- quickBamFlagSummary("./data/example/calls.bam", main.groups.only=TRUE) 
data1 <- quickBamFlagSummary("./data/example/has_reference.bam", main.groups.only=TRUE) 
flag1 <- scanBamFlag(isFirstMateRead=TRUE, 
                     isSecondMateRead=FALSE,
                     isDuplicate=FALSE, 
                     isNotPassingQualityControls=FALSE) 
param1 <- ScanBamParam(flag=flag1, what="seq") 
data1 <- readGAlignments("./data/example/has_reference.bam", use.names=TRUE, param=param1) 
class(data1)
data1


signalFiles <- c(system.file("extdata",
                             "./data/example/has_reference.bam",
                             package = "profileplyr"))
for (i in seq_along(signalFiles)){
  indexBam(signalFiles[i])
}


h5_file= H5Fopen("./data/example/PAG65784_pass_f306681d_16a70748_999.fast5")

hdump=h5dump(h5_file,load=FALSE)

fast5_file <- "./data/example/PAG65784_pass_f306681d_16a70748_999.fast5"
fast5_data <- h5read(fast5_file, "/read_0000b1ad-fdaf-49e6-bc11-cbe93270e3a3/Raw/Signal")
