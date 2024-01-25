cat('',file='seed_index.txt',append=F)

for(i in c(1:100000)){
seed=as.integer(ceiling(runif(1L)*2^15))

cat(paste(seed,'\n'),file='seed_index.txt',append=T)
}
