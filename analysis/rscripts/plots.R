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

# of analysis
dof = read_csv("../data/combined_of_status.csv")
colnames(dof) = c("Weight","Of", "Normalized")

dof %<>% 
  mutate_if(is.logical,funs(factor(.))) %>%
  mutate(Of = fct_recode(Of,"\"some of\""="TRUE","other \"of\""="FALSE"),
         Normalized = fct_recode(Normalized,"normalized"="TRUE","raw"="FALSE")) 
dof$Of = factor(dof$Of, levels = c("\"some of\"", "other \"of\""), ordered=T)

means = dof %>%
  group_by(Of,Normalized) %>%
  summarise(Mean = mean(Weight),CILow=ci.low(Weight),CIHigh=ci.high(Weight)) %>%
  ungroup() %>%
  mutate(YMin=Mean-CILow,YMax=Mean+CIHigh)
dodge = position_dodge(.9)

of_plot = ggplot(means, aes(x=Normalized,y=Mean,fill=Of)) +
  geom_errorbar(aes(ymin=YMin,ymax=YMax),width=0.2) +
  geom_point(color="black",pch=21,size=3) +
  scale_fill_manual(name="",values=cbPalette[c(5,7)]) + 
  ylab("Mean Attention Weight")  + theme_bw() + theme(legend.position = "bottom") + xlab("")
ggsave("../graphs/of_analysis.pdf",plot=of_plot, width=3,height=3)

# qualitative analysis (subjecthood, partitive, modification)
dqual = read_csv("../data/800_artificial.csv")
colnames(dqual) = c("ID","Sentence","Preds","Partitive","Prenominal","Postnominal","Subjecthood","Passive","Modification")
dqual %<>% mutate(Sentence=fct_reorder(Sentence,Preds)) %>%
  mutate_if(is.logical,funs(factor(.))) %>%
  mutate(Partitive = fct_recode(Partitive,partitive="TRUE","non-partitive"="FALSE"),
         Subjecthood = fct_recode(Subjecthood,subject="TRUE",other="FALSE"),
         Modification = fct_recode(Modification,"modified"="TRUE","unmodified"="FALSE"))


dqual$Subjecthood = factor(dqual$Subjecthood, levels=c("subject", "other"), ordered = T)
dqual$Modification = factor(dqual$Modification, levels=c("modified", "unmodified"), ordered = T)
dqual$Partitive = factor(dqual$Partitive, levels=c("partitive", "non-partitive"), ordered = T)



dqual_all = dqual %>%
  mutate(Predictor="Partitive", Value=Partitive)

dqual_all = rbind(dqual_all, 
                  dqual %>%
                  mutate(Predictor="Modification", Value=Modification))

dqual_all = rbind(dqual_all, 
                  dqual %>%
                    mutate(Predictor="Subjecthood", Value=Subjecthood))

part_means = dqual %>%
  group_by(Partitive) %>%
  summarise(Prediction = mean(Preds),CILow=ci.low(Preds),CIHigh=ci.high(Preds)) %>%
  ungroup() %>%
  mutate(YMin=Prediction-CILow,YMax=Prediction+CIHigh) %>%
  mutate(Predictor="Partitive") %>% rename(Value=Partitive)

mod_means = dqual %>%
  group_by(Modification) %>%
  summarise(Prediction = mean(Preds),CILow=ci.low(Preds),CIHigh=ci.high(Preds)) %>%
  ungroup() %>%
  mutate(YMin=Prediction-CILow,YMax=Prediction+CIHigh) %>%
  mutate(Predictor="Modification") %>% rename(Value=Modification)
  
subj_means = dqual %>%
  group_by(Subjecthood) %>%
  summarise(Prediction = mean(Preds),CILow=ci.low(Preds),CIHigh=ci.high(Preds)) %>%
  ungroup() %>%
  mutate(YMin=Prediction-CILow,YMax=Prediction+CIHigh) %>%
  mutate(Predictor="Subjecthood") %>% rename(Value=Subjecthood)

means = rbind(part_means, mod_means, subj_means)
means$Predictor = factor(means$Predictor, levels=c("Partitive", "Subjecthood", "Modification"), ordered=T)


dodge = position_dodge(.9)
jitter = position_jitter(width = .2)

qualplot = ggplot(means, aes(x=Value,y=Prediction)) +
  geom_point(data=dqual_all,aes(y=Preds),alpha=.1, color=cbPalette[2], position=jitter,size=2) +
  #geom_errorbar(aes(ymin=YMin,ymax=YMax),position=dodge,width=0, size=1) +
  geom_point(position=dodge,color="black", fill=cbPalette[6], pch=21,size=7) +
  scale_fill_manual(values=cbPalette[c(1,2)]) +
  scale_color_manual(values=cbPalette[c(1,2)]) + theme_bw() +
  theme(legend.position = "bottom", axis.title.x = element_blank(), axis.text.x = element_text(size=9), strip.text = element_text(size=12)) +
  facet_wrap(~Predictor, scales = "free_x")
ggsave("../graphs/800_qualitative.pdf", plot=qualplot, width=5,height=4)

means = dqual %>%
  group_by(Modification) %>%
  summarise(Prediction = mean(Preds),CILow=ci.low(Preds),CIHigh=ci.high(Preds)) %>%
  ungroup() %>%
  mutate(YMin=Prediction-CILow,YMax=Prediction+CIHigh)
dodge = position_dodge(.9)

ggplot(means, aes(x=Modification,y=Prediction)) + 
  geom_errorbar(aes(ymin=YMin,ymax=YMax),position=dodge,width=0.2) +
  geom_point(position=dodge,color="black",pch=21,size=3) +
  theme_bw() +ylim(4.5,5.5)
ggsave("../graphs/modification.png",width=6,height=4)


# attention weight analysis (by position)
dattn = read_csv("../data/attn_by_pos_subj_30.csv")
#dattn = read_csv("../data/attn_by_pos_subj_30_withSome.csv")
colnames(dattn) = c("Subjecthood","Position","Weight")
dattn %<>% 
  mutate(Subjecthood = fct_recode(Subjecthood,subject="yes",other="no"))

dattn$Subjecthood = factor(dattn$Subjecthood, levels=c("subject", "other"), ordered = T)

means = dattn %>%
  group_by(Position, Subjecthood) %>%
  summarise(Mean = mean(Weight), CILow=ci.low(Weight),CIHigh=ci.high(Weight)) %>% 
  ungroup() %>%
  mutate(YMin=Mean-CILow,YMax=Mean+CIHigh)
dodge = position_dodge(.9)

attn_plot = ggplot(means, aes(x=Position,y=Mean,fill=Subjecthood)) + 
  geom_errorbar(aes(ymin=YMin,ymax=YMax),position=dodge,width=0) +
  geom_point(position=dodge,color="black",pch=21,size=2) +
  scale_fill_manual(values=cbPalette[c(1,2)]) + 
  scale_color_manual(values=cbPalette[c(1,2)]) +
  xlab("Position in Utterance") +
  ylab("Mean Attention Weight") +
  theme_bw() + theme(legend.position = "bottom")
ggsave("../graphs/avgAttnNaturalSubj30.pdf", plot=attn_plot, width=3,height=3)
#ggsave("../graphs/avgAttnNaturalSubj30_withSome.png",width=6,height=4)

# all learning curves
dmod = read_csv("../data/all_learning_curves.csv")
dmod %<>% mutate_if(is.character,funs(factor(.))) %>%
  mutate(embedding_method=as.factor(case_when(
    lstm_attn == "TRUE" ~ "LSTM+Attention", lstm_attn == "FALSE" ~ "LSTM"  
    ))) %>%
  mutate(with_context = fct_recode(context,yes="Contextual",no="Single"))

levels(dmod$embedding) <- c("BERT-base", "BERT-large", "ELMo", "GloVe")

ggplot(subset(dmod, dmod$epoch <= 200), aes(x=epoch,y=avg_val_corr,color=embedding_method, linetype=with_context)) +
  # geom_point() +
  geom_line(size=.5) + ylim(.4,.8) + ylab("Average validation correlation") +
  scale_linetype(drop=F) + labs(linetype="Preceding context") +
  scale_color_manual(name="Sentence representation",values=cbPalette[c(5,6)],drop=F) +
  scale_x_continuous(name="Epoch", breaks=seq(0,200,by=40)) + 
  facet_wrap(~embedding,drop=F,nrow=1) +
  guides(color = guide_legend(order=1),
         linetype = guide_legend(order=2)) + theme_bw() + theme(legend.position="bottom")
ggsave("../graphs/val_corr.png",width=7,height=3)