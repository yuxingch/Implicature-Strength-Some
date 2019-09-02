reshapeData <- function(d)
{
  d$Trial1 = paste(d$Gender0,d$Speaker0,d$Quantifier0,d$Prior0,d$Slider00,d$Slider01,d$Slider02,d$Slider03,d$Slider04)
  d$Trial2 = paste(d$Gender1,d$Speaker1,d$Quantifier1,d$Prior1,d$Slider10,d$Slider11,d$Slider12,d$Slider13,d$Slider14)
  d$Trial3 = paste(d$Gender2,d$Speaker2,d$Quantifier2,d$Prior2,d$Slider20,d$Slider21,d$Slider22,d$Slider23,d$Slider24)
  d$Trial4 = paste(d$Gender3,d$Speaker3,d$Quantifier3,d$Prior3,d$Slider30,d$Slider31,d$Slider32,d$Slider33,d$Slider34)  
  return(d) 
} 

getQUD <- function(qud) {
  #print(qud)
  if (length(grep("How many", qud)) > 0) {
    return("HowMany?")
  } else {
    if (length(grep("all", qud)) > 0) {
      return("All?")
    } else {
      if (length(grep("Are any", qud)) > 0) {
        return("Any?")
      } else {
        return("ERROR!")
      }
    }
  }
}

myCenter <- function(x) {
  if (is.numeric(x)) { return(x - mean(x)) }
  if (is.factor(x)) {
    x <- as.numeric(x)
    return(x - mean(x))
  }
  if (is.data.frame(x) || is.matrix(x)) {
    m <- matrix(nrow=nrow(x), ncol=ncol(x))
    colnames(m) <- paste("c", colnames(x), sep="")
    for (i in 1:ncol(x)) {
      if (is.factor(x[,i])) {
        y <- as.numeric(x[,i])
        m[,i] <- y - mean(y, na.rm=T)
      }
      if (is.numeric(x[,i])) {
        m[,i] <- x[,i] - mean(x[,i], na.rm=T)
      }
    }
    return(as.data.frame(m))
  }
}

se <- function(x)
{
  y <- x[!is.na(x)] # remove the missing values, if any
  sqrt(var(as.vector(y))/length(y))
}

zscore <- function(x){
  ## Returns z-scored values
  x.mean <- mean(x)
  x.sd <- sd(x)
  
  x.z <- (x-x.mean)/x.sd
  
  return(x.z)
}

zscoreByGroup <- function(x, groups){ 
  #Compute zscores within groups
  out <- rep(NA, length(x))
  
  for(i in unique(groups)){
    out[groups == i] <- zscore(x[groups == i])
  }
  return(out)
}

## for bootstrapping 95% confidence intervals
library(bootstrap)
theta <- function(x,xdata,na.rm=T) {mean(xdata[x],na.rm=na.rm)}
ci.low <- function(x,na.rm=T) {
  mean(x,na.rm=na.rm) - quantile(bootstrap(1:length(x),1000,theta,x,na.rm=na.rm)$thetastar,.025,na.rm=na.rm)}
ci.high <- function(x,na.rm=T) {
  quantile(bootstrap(1:length(x),1000,theta,x,na.rm=na.rm)$thetastar,.975,na.rm=na.rm) - mean(x,na.rm=na.rm)}

library(Hmisc)

createLatexTableLinear = function(m, predictornames=c())
{
  coefs = m
  
  coefs[,1] = round(coefs[,1],digits=2)
  coefs[,2] = round(coefs[,2],digits=2)
  coefs[,4] = round(coefs[,4],digits=1)
  coefs$P = ifelse(coefs[,5] > .05, paste(">",round(coefs[,5],digits=2),sep=""), ifelse(coefs[,5] < .0001, "\\textbf{<.0001}", ifelse(coefs[,5] < .001,"\\textbf{<.001}", ifelse(coefs[,5] < .01, "\\textbf{<.01}", "\\textbf{<.05}"))))
  head(coefs)
  coefs[,3] = NULL
  coefs[,4] = NULL  
  colnames(coefs) = c("Coef $\\beta$","SE($\\beta$)", "\\textbf{t}","$p$")
  coefs
  
  if (length(predictornames > 0))
  {
    prednames = data.frame(PName=row.names(coefs),NewNames=predictornames)#c("Intercept","Above BP","Accessible head noun.0","Accessible head noun.1","Grammatical function (subject).0","Grammatical function (subject).1","Animate head noun.0","Animate head noun.1","Head predictability.0","Head predictability.1","Constituent complexity.0","Constituent complexity.1","IPre.0","IPre.1","IHead.0","IHead.1","Head type (count).0","Head type (count).1"))
  } else {
    prednames = data.frame(PName=row.names(coefs),NewNames=row.names(coefs))		
  }
  
  row.names(coefs) = prednames$NewNames[prednames$PName == row.names(coefs)]
  
  latex(coefs,file="",title="",table.env=TRUE,booktabs=TRUE)
}
