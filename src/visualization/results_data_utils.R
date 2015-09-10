##
## Useful commands for grabbing raw tsv files of the same type
## and concatting them together. This makes viz and analysis
## of the data a breeze.
##

library(stringr)
library(rjson)

RESULTS.DIR <- "../../results/"

makeDirIfMissing <- function(filepath) {
    d <- dirname(filepath)
    dir.create(d, showWarnings=FALSE)
}

capitalizeFirstInitial <- function(str) {
    words <- strsplit(str, " ")[[1]]
    first.word <- words[1]
    cap.first.word <- paste(
        toupper(substring(first.word,1,1)),
        substring(first.word, 2),
        sep=""
    )
    paste(c(cap.first.word, words[-1]), collapse=" ")
}

pluckParamFromFilename <- function(filenames, param) {
  reg.match <- str_match(filenames, sprintf("_?%s=([^_\\.]+)_?", param))
  reg.match[,2]
}

getParamID <- function(param, default=NULL) {
    if (is.null(default)) {
       default <- ".*"
    }
    if (is.null(param)) {
        return(default)
    } else {
        return(param)
    }
}

# If you make a arg have NULL, it just will let it
# have any value, but require it present
#
# The only "special" arguments are
# prefix="julia" [this doesnt have an `=` separator]
# ext="tsv" [this is the file extension]
getMatchingFiles <- function(dir=NA, ...) {
    # order of kwargs is important!!
    kwargs <- as.list(substitute(list(...)))[-1L]
    file.prefix <- getParamID(eval(kwargs$prefix), "julia")
    kwargs$prefix <- NULL  # erase it
    file.ext <- getParamID(eval(kwargs$ext), "tsv")
    kwargs$ext <- NULL

    kwarg.chunks <- sapply(names(kwargs), function(kwarg) {
        kwval <- eval(kwargs[[kwarg]])
        if (is.null(kwval)) {
          kwval <- ".*"
        }
        sprintf("%s=%s", kwarg, kwval)
    })

    all.chunks <- c(file.prefix, kwarg.chunks)
    file.base.re <- paste(all.chunks, collapse="_.*")
    # add file extension
    matching.re <- paste0(file.base.re, "(_.*)?\\.", file.ext)
    print(paste("[Regex]:", matching.re))
    # get directory to look in
    search.dir <- ifelse(
        is.na(dir),
        file.path(RESULTS.DIR, "misc", "data"),
        file.path(RESULTS.DIR, dir, "data")
    )
    data.files <- list.files(search.dir)
    matching.files <- grep(matching.re, data.files, value=T)
    num.matches <- length(matching.files)
    print(sprintf("# matched files: %d", num.matches))
    if (length(matching.files) == 0) {
      return(c())
    }
    paste(search.dir, matching.files, sep="/")
}

loadDataFileTSV <- function(
    matching.file,
    header=FALSE,
    inject.params=c(),
    header.names=c()
) {
    tmp.df <- read.csv(matching.file, header=header, sep="\t")
    if (length(header.names)) {
        colnames(tmp.df) <- header.names
    }
    for (param in inject.params) {
        tmp.df[, param] <- pluckParamFromFilename(matching.file, param)
    }
    tmp.df
}

loadDataFileJSON <- function(filename) {
    fromJSON(file=filename)
}

overwriteDataFile <- function(
    matching.file,
    df,
    header=FALSE
) {
    write.table(df, file=matching.file, col.names=header, row.names=F, sep="\t", quote=F)
}

concatData <- function(
    header=FALSE,
    inject.params=c(),
    header.names=c(),
    dir=NA,
    ...
) {
    matching.files <- getMatchingFiles(dir=dir, ext="tsv", ...)
    if (length(matching.files) == 0) {
      stop("No matching files")
    }
    df <- data.frame()
    for (matching.file in matching.files) {
        tmp.df <- loadDataFileTSV(matching.file,
                                   header=header,
                                   inject.params=inject.params,
                                   header.names=header.names)
        df <- rbind(df, tmp.df)
    }
    df
}

concatDataList <- function(
    dir=NA,
    ...
) {
    matching.files <- getMatchingFiles(dir=dir, ext="json", ...)
    if (length(matching.files) == 0) {
      stop("No matching files")
    }
    ll <- list()
    for (matching.file in matching.files) {
        tmp.list <- loadDataFileJSON(matching.file)
        ll <- append(ll, list(tmp.list))
    }
    ll
}

number_ticks <- function(n, logged=TRUE) {
    function (limits) {
        b <- pretty(limits, n)
        if (logged) {
            b <- b[b > 0]
        }
        b
    }
}

### Some examples

# foo <- concatData(datatype="gfunc")
# concatData(n=2000, header.names=c('n', 'objValue', 'time'), inject.params=('seed'))

#### Replace summary data
## all.filenames <- getMatchingFiles(problemtype=NULL)
## col.order <- c('dist', 'n', 'problemtype', 'p', 'seed', 'bound', 'time')
## for (matching.file in all.filenames) {
##     df <- loadDataFile(
##         matching.file,
##         header.names=c('n', 'bound', 'time'),
##         inject.params=c('dist', 'problemtype', 'p', 'seed')
##     )
##     df <- df[, col.order]
##     overwriteDataFile(matching.file, df)
## }

## ## Replace gfunc data
## all.filenames <- getMatchingFiles(problemtype=NULL, datatype="gfunc")
## col.order <- c('dist', 'n', 'problemtype', 'p', 'seed', 'X', 'g', 'gPrime')
## for (matching.file in all.filenames) {
##     df <- loadDataFile(
##         matching.file,
##         header.names=c('n', 'X', 'g', 'gPrime'),
##         inject.params=c('dist', 'problemtype', 'p', 'seed')
##     )
##     df <- df[, col.order]
##     overwriteDataFile(matching.file, df)
## }
