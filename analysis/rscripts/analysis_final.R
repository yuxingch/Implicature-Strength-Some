library(Hmisc)
library(gridExtra)
library(MuMIn)
library(tidyverse)
library(magrittr)

# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

# load the original SI data
load("../data/complete_md.RData")
source("helpers.R")

# load the model evaluation data
dmod = read_csv("../data/model_performance_eval.csv")

# load qualitative model prediction dataset
dqual = read_csv("../data/qualitative_model_predictions.csv")

# To run the regression reported in Degen 2015:
centered = cbind(md, myCenter(md[,c("StrengthSome","logSentenceLength","Pronoun","BinaryGF","InfoStatus","DAModification","Modification","Partitive","redInfoStatus","numDA")]))

m.random = lmer(Rating ~  (1|workerid), data=centered)
summary(m.random)

m.fixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=centered)
summary(m.fixed)

anova(m.random,m.fixed)


m = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid) + (0 + cPartitive|workerid) + (0 + cStrengthSome|workerid) + (0 + credInfoStatus|workerid) + (0 + cBinaryGF|workerid) + (0+cModification|workerid) + (0 + cPartitive:cStrengthSome|workerid) + (1|Item), data=centered)
summary(m)
msummary = summary(m)
coefs = as.data.frame(msummary$coefficients)
summary(coefs)
head(coefs)

createLatexTableLinear(coefs,predictornames=c("Intercept","Partitive","Strength","Linguistic mention","Topicality","Modification","Sentence length","Partitive:Strength","Linguistic mention:Topicality","Linguistic mention:Modification","Topicality:Modification","Linguistic mention:Topicality:Modification"))


# Add Yuxing's vector representations of each item
vheaders = paste("V",seq(1,100),sep="")
embs = read_tsv("../data/embs_single.csv") %>%
  separate(Vector_Representation,vheaders,sep=",") %>%
  rename(Item = Item_ID)

tmp = embs %>%
  mutate_if(is.character,as.numeric) %>%
  mutate(VectorSum = rowSums(.[2:101]))
tmp$Item = embs$Item
head(embs)

d = md %>%
  left_join(tmp) 
head(d,20)

write.csv(d, "../data/some_data_withembs.tsv",row.names=F)

# Add Yuxing's predictions
preds = read_tsv("../data/predictions.csv") %>%
  rename(Item = Item_ID)

d = d %>%
  left_join(preds)
  
# run the models to compare vectorsum vs hand mined features
centered = cbind(d, myCenter(d[,c("StrengthSome","logSentenceLength","Pronoun","BinaryGF","InfoStatus","DAModification","Modification","Partitive","redInfoStatus","numDA")]))

m.vectorpred = lmer(Rating ~ predicted + (1|workerid), data=centered)
summary(m.vectorpred) 

m.vector = lmer(formula(paste("Rating ~ ",paste("V",seq(1,100),sep="",collapse="+"),"+ (1|workerid)")), data=centered)
summary(m.vector) 

m.fixedandvector = lmer(formula(paste("Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength +",paste("V",seq(1,100),sep="",collapse="+"),"+ (1|workerid)")), data=centered)
summary(m.fixedandvector) 

m.fixedandvectorpred = lmer(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=centered)
summary(m.fixedandvectorpred)

m.fixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=centered)
summary(m.fixed)

anova(m.fixed,m.fixedandvectorpred) # chi squared (1) = 3086.7, p < .0001
anova(m.vectorpred,m.fixedandvectorpred) # chi squared (11) = 74.6, p < .0001

centered$PredictedVector = fitted(m.vector)
centered$PredictedVectorPred = fitted(m.vectorpred)
centered$PredictedFixed = fitted(m.fixed)
centered$PredictedFV = fitted(m.fixedandvector)
centered$PredictedFVP = fitted(m.fixedandvectorpred)

agr = centered %>%
  group_by(Item,Sentence) %>%
  summarise(MeanEmpirical = mean(Rating), MeanPredictedVector = mean(PredictedVector),MeanPredictedVectorPred = mean(PredictedVectorPred),MeanPredictedFixed = mean(PredictedFixed), MeanPredictedFV = mean(PredictedFV),MeanPredictedFVP = mean(PredictedFVP)) 

write.csv(agr,"../data/some_means_emppred.csv",row.names=F)
 
cor(agr$MeanPredictedVector, agr$MeanEmpirical) # corr: .62
cor(agr$MeanPredictedFixed, agr$MeanEmpirical) # corr: .66
cor(agr$MeanPredictedFV, agr$MeanEmpirical) # corr: .73
cor(agr$MeanPredictedVectorPred, agr$MeanEmpirical) # corr: .90
cor(agr$MeanPredictedFVP, agr$MeanEmpirical) # corr: .91

r.squaredGLMM(m.vector)
r.squaredGLMM(m.vectorpred)
r.squaredGLMM(m.fixed)
r.squaredGLMM(m.fixedandvector)
r.squaredGLMM(m.fixedandvectorpred)


ggplot(agr, aes(x=MeanPredictedFixed,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,7) +
  ylim(0,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_fixed.pdf",width=5,height=4)

ggplot(agr, aes(x=MeanPredictedVector,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,7) +
  ylim(0,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_vector.pdf",width=5,height=4)

ggplot(agr, aes(x=MeanPredictedFV,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,7) +
  ylim(0,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_fv.pdf",width=5,height=4)

ggplot(agr, aes(x=MeanPredictedVectorPred,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,7) +
  ylim(0,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_vectorpred.pdf",width=5,height=4)

ggplot(agr, aes(x=MeanPredictedFVP,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,7) +
  ylim(0,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_fvp.pdf",width=5,height=4)


################################
# Neural model evaluation ----
################################

dmod %<>% mutate_if(is.character,funs(factor(.))) %>%
  mutate(embedding_method=as.factor(case_when(
    attention == "Attention" ~ "LSTM+attention",
    lstm == "not LSTM" ~ "average",
    TRUE ~ "LSTM"
  ))) %>%
  mutate(Context = fct_recode(context,yes="Contextual",no="Single"))
summary(dmod)


ggplot(dmod, aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)]) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=40)) +
  facet_wrap(~word_embedding) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance.pdf",width=7,height=3)

# full xprag correlation plot
ggplot(dmod %>% filter(epoch <= 80), aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)]) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=20)) +
  facet_wrap(~word_embedding) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance_epoch80.pdf",width=7,height=3)

# xprag correlation plot 1
ggplot(subset(dmod,dmod$embedding_method == "average" & dmod$word_embedding == "GloVe" & Context == "no"), aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_linetype(drop=F) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)],drop=F) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=20)) +
  facet_wrap(~word_embedding,drop=F) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance_epoch80_1.pdf",width=7,height=3)

# xprag correlation plot 2
ggplot(subset(dmod,dmod$embedding_method == "average" & dmod$word_embedding == "GloVe"), aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_linetype(drop=F) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)],drop=F) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=20)) +
  facet_wrap(~word_embedding,drop=F) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance_epoch80_2.pdf",width=7,height=3)

# xprag correlation plot 3
ggplot(subset(dmod,dmod$embedding_method == "average" & dmod$epoch <= 200), aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_linetype(drop=F) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)],drop=F) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=20)) +
  facet_wrap(~word_embedding,drop=F) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance_epoch80_3.pdf",width=7,height=3)

# xprag correlation plot 4
ggplot(subset(dmod,dmod$embedding_method == "average" & dmod$epoch <= 80), aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_linetype(drop=F) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)],drop=F) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=20)) +
  facet_wrap(~word_embedding,drop=F) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance_epoch80_4.pdf",width=7,height=3)

# xprag correlation plot 5
ggplot(subset(dmod,dmod$embedding_method != "LSTM+attention" & dmod$epoch <= 80), aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_linetype(drop=F) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)],drop=F) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=20)) +
  facet_wrap(~word_embedding,drop=F) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance_epoch80_5.pdf",width=7,height=3)

# xprag correlation plot 6
ggplot(subset(dmod, dmod$epoch <= 80), aes(x=epoch,y=r,color=embedding_method, linetype=Context)) +
  # geom_point() +
  geom_line(size=.8) +
  ylim(-.15,.8) +
  scale_linetype(drop=F) +
  scale_color_manual(name="Embedding method",values=cbPalette[c(1,2,7)],drop=F) +
  scale_x_continuous(name="Epoch", breaks=seq(0,300,by=20)) +
  facet_wrap(~word_embedding,drop=F) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2))
ggsave("../graphs/model_performance_epoch80_6.pdf",width=7,height=3)

max(dmod$r)

# qualitative model predictions (ELMo+LSTM+Attn--Single sentence)
colnames(dqual) = c("Sentence","Prediction","Partitive","Prenominal_modification","Postnominal_modification","Subjecthood")

dqual %<>% mutate(Sentence=fct_reorder(Sentence,Prediction)) %>%
  mutate_if(is.logical,funs(factor(.))) %>%
  mutate(Partitive = fct_recode(Partitive,partitive="TRUE","non-partitive"="FALSE"),
         Subjecthood = fct_recode(Subjecthood,subject="TRUE",other="FALSE"),
         Modification = case_when(
           Prenominal_modification == "TRUE" & Postnominal_modification == "TRUE" ~ "pre- and postnominal modification",
           Prenominal_modification == "TRUE" & Postnominal_modification == "FALSE" ~ "prenominal modification",
           Prenominal_modification == "FALSE" & Postnominal_modification == "TRUE" ~ "postnominal modification",
           Prenominal_modification == "FALSE" & Postnominal_modification == "FALSE" ~ "no modification"
         ))

ggplot(dqual, aes(x=Sentence,y=Prediction)) +
  geom_point() +
  coord_flip()

ggplot(dqual, aes(x=Sentence,y=Prediction,color=Partitive,shape=Subjecthood)) +
  geom_point(size=3) +
  coord_flip() +
  scale_color_manual(values=cbPalette[c(5,7)]) +
  facet_wrap(~Modification,ncol = 1,scales="free_y")
ggsave("../graphs/qualitative_predictions.pdf",width=8,height=8)  

means = dqual %>%
  group_by(Partitive,Subjecthood) %>%
  summarise(Mean = mean(Prediction),CILow=ci.low(Prediction),CIHigh=ci.high(Prediction)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,YMax=Mean+CIHigh)
dodge = position_dodge(.9)

ggplot(means, aes(x=Partitive,y=Mean,fill=Subjecthood)) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),position=dodge,width=0) +
  geom_point(position=dodge,color="black",pch=21,size=3) +
  geom_point(data=dqual,aes(y=Prediction,color=Subjecthood),alpha=.4,position=dodge,size=2) +
  scale_fill_manual(values=cbPalette[c(5,7)]) +
  scale_color_manual(values=cbPalette[c(5,7)])
ggsave("../graphs/qualitative_prediction_means.pdf",width=5,height=3)  

dqual_partsubj = dqual %>%
  filter(Modification == "no modification")

ggplot(dqual_partsubj, aes(x=Sentence,y=Prediction,color=Partitive,shape=Subjecthood)) +
  geom_point(size=4) +
  scale_color_manual(values=cbPalette[c(5,7)]) +
  ylim(2.5,7.5) +
  coord_flip() +
  theme(axis.text.y=element_text(size=12))

ggsave("../graphs/qualitative_prediction_means_partsubj.pdf",width=7,height=2.5)  

# scatterplots of model predictions against empirical means
d_elmo_lstmattn_context = read_tsv("../data/implicature_strength_predictions/ELMo_Contextual_LSTM_Attn/all_preds_epoch70_elmo_contextual_lstm_attn.csv") %>%
  mutate(Dataset = fct_recode(label,evaluation="eval",training="train"))

cors = d_elmo_lstmattn_context %>%
  group_by(Dataset) %>%
  summarise(Correlation = round(cor(predicted,original_mean),2))

ggplot(d_elmo_lstmattn_context,aes(x=predicted,y=original_mean,color=Dataset)) +
  geom_point() +
  geom_text(data=cors, aes(label=Correlation),x=1.5,y=6.5,color="black") +
  guides(color=F) +
  xlim(1,7) +
  ylim(1,7) +
  ylab("Empirical mean") +
  xlab("Predicted mean") +
  facet_wrap(~Dataset)
ggsave(file="../graphs/scatter_elmo_lstmattn_context.pdf",width=5.5,height=3)

# compare the best lstim+attention model to hand-mined model predictions
preds = d_elmo_lstmattn_context %>%
  select(Item_ID,predicted,Dataset)
colnames(preds) = c("Item","PredictedMean","Dataset")

md %<>%
  left_join(preds, by=c("Item"))

library(lmerTest)

# To run the regression reported in Degen 2015:
centered = cbind(md, myCenter(md[,c("StrengthSome","logSentenceLength","Pronoun","BinaryGF","InfoStatus","DAModification","Modification","Partitive","redInfoStatus","numDA","PredictedMean")]))

m.fixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid) + (0 + cPartitive|workerid) + (0 + cStrengthSome|workerid) + (0 + credInfoStatus|workerid) + (0 + cBinaryGF|workerid) + (0+cModification|workerid) + (0 + cPartitive:cStrengthSome|workerid), data=centered %>% filter())
summary(m.fixed)

m.neural = lmer(Rating ~ cPredictedMean + (1|workerid), data=centered)
summary(m.neural)

m.fixed.neural = lmer(Rating ~ cPredictedMean + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid) + (0 + cPartitive|workerid) + (0 + cStrengthSome|workerid) + (0 + credInfoStatus|workerid) + (0 + cBinaryGF|workerid) + (0+cModification|workerid) + (0 + cPartitive:cStrengthSome|workerid) , data=centered)
fnsummary = summary(m.fixed.neural)

coefs = as.data.frame(fnsummary$coefficients)
summary(coefs)
head(coefs)

anova(m.neural,m.fixed.neural)

createLatexTableLinear(coefs,predictornames=c("Intercept","Neural model score","Partitive","Strength","Linguistic mention","Subjecthood","Modification","Sentence length","Partitive:Strength","Linguistic mention:Subjecthood","Linguistic mention:Modification","Subjecthood:Modification","Linguistic mention:Subjecthood:Modification"))

centered$PredictedFixed = fitted(m.fixed)
centered$PredictedNeural = fitted(m.neural)
centered$PredictedFixedNeural = fitted(m.fixed.neural)

agr = centered %>%
  group_by(Item,Sentence,Dataset) %>%
  summarise(MeanEmpirical = mean(Rating), MeanPredictedFixed = mean(PredictedFixed),MeanPredictedNeural = mean(PredictedNeural),MeanPredictedFixedNeural = mean(PredictedFixedNeural))

ggplot(agr %>% filter(Dataset=="evaluation"), aes(x=MeanPredictedNeural,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,7) +
  ylim(0,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_neural.pdf",width=3,height=2.5)

ggplot(agr %>% filter(Dataset=="evaluation"), aes(x=MeanPredictedFixed,y=MeanEmpirical)) +
  geom_point() +
  geom_smooth(method="lm") +
  xlim(0,7) +
  ylim(0,7) +
  ylab("Empirical rating") +
  xlab("Predicted rating")
ggsave("../graphs/model_fit_fixed.pdf",width=3,height=2.5)
