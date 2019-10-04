library(tidyverse)
library(brms)
library(lme4)
library(lmerTest)
library(stringr)


#setwd("analysis/rscripts/")

setwd("~/Dropbox/Uni/RA/implicature-strength/Research/analysis/rscripts/")
theme_set(theme_bw())

# color-blind-friendly palette
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") 

# load the original SI data
load("../data/complete_md.RData")
source("helpers.R")
source("stan_utility.R")

# To run the regression reported in Degen 2015:
centered = cbind(md, myCenter(md[,c("StrengthSome","logSentenceLength","Pronoun","BinaryGF","InfoStatus","DAModification","Modification","Partitive","redInfoStatus","numDA")]))

# LMER model
m.fixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=centered)
summary(m.fixed)

# BRMS model
bm.fixed = brm(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=centered)
summary(bm.fixed)


################################################
#  Models estimated only from evaluation data  #
################################################


# Add Yuxing's predictions
preds = read_tsv("../../runs.eval/bert_large_lstm_eval/Preds/test_preds_rating_epoch190.csv") %>%
  rename(Item = Item_ID)

d = centered %>%
  right_join(preds)

# LMER model with random by-subject intercepts and model predictions 
m.evalvectorpred = lmer(Rating ~ predicted + (1|workerid), data=d)
summary(m.evalvectorpred) 

# BRMS model with random by-subject intercepts and model predictions 
bm.evalvectorpred = brm(Rating ~ predicted + (1|workerid), data=d)
summary(bm.evalvectorpred) 

# LMER model with random by-subject intercepts, fixed effects, and model predictions 
m.evalfixedandvectorpred = lmer(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(m.evalfixedandvectorpred)

# BRMS model with random by-subject intercepts, fixed effects, and model predictions 
bm.evalfixedandvectorpred = brm(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(bm.evalfixedandvectorpred)
stanplot(bm.evalfixedandvectorpred)

# LMER model with random by-subject intercepts and fixed effects
m.evalfixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(m.evalfixed)

# BRMS model with random by-subject intercepts and fixed effects
bm.evalfixed = brm(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(bm.evalfixed)
stanplot(bm.evalfixed)

# compare coefficients of models with and without NN model predictions

bm.evalfixedandvectorpred.est = bm.evalfixedandvectorpred$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "fixed+nn predictions")

bm.evalfixed.est = bm.evalfixed$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "fixed")


estimates = rbind(bm.evalfixedandvectorpred.est, bm.evalfixed.est)
estimates %>% ggplot(aes(y=str_replace(Parameter, "b_", ""), x=mu, col=model)) + 
  geom_point(size=3) + geom_point(size=2.0, col="white", alpha=0.6) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh), height=.1) +
  xlab("coefficient estimate") +
  ylab("Parameter") +
  theme(legend.position = "bottom")


# simulated estimates of P(beta_fixed < beta_nn) for all fixed effects

bm.evalfixed_df = bm.evalfixed$fit %>% 
  as.data.frame()

params = colnames(bm.evalfixed_df)[grepl("b_", colnames(bm.evalfixed_df))]

bm.evalfixed_df = bm.evalfixed_df %>%
  select(params)

bm.evalfixedandvectorpred_df = bm.evalfixedandvectorpred$fit %>%
  as.data.frame() %>%
  select(params)


n = 1000000
fixed_sample = bm.evalfixed_df %>% sample_n(size = n, replace = T)
nn_sample = bm.evalfixedandvectorpred_df %>% sample_n(size = n, replace = T)
sim_results = data.frame(fixed_sample < nn_sample)


sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise("P(beta_fixed < beta_nn)" = mean(value))



################################################
#  Cross-validation models                     #
################################################


# Add Yuxing's predictions


preds = data.frame()

for (i in 0:5) {
  preds_fold = read_tsv(paste("../../runs.eval/bert_large_lstm_eval_split_", i, "/Preds/test_preds_rating_epoch190.csv", sep="")) %>%
    rename(Item = Item_ID)
  preds = rbind(preds, preds_fold)
}
  
  

d = centered %>%
  right_join(preds)

# LMER model with random by-subject intercepts and model predictions 
m.cvvectorpred = lmer(Rating ~ predicted + (1|workerid), data=d)
summary(m.cvvectorpred) 

# BRMS model with random by-subject intercepts and model predictions 
bm.cvvectorpred = brm(Rating ~ predicted + (1|workerid), data=d)
summary(bm.cvvectorpred) 

# LMER model with random by-subject intercepts, fixed effects, and model predictions 
m.cvfixedandvectorpred = lmer(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(m.cvfixedandvectorpred)

# BRMS model with random by-subject intercepts, fixed effects, and model predictions 
bm.cvfixedandvectorpred = brm(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(bm.cvfixedandvectorpred)
stanplot(bm.cvfixedandvectorpred)

# LMER model with random by-subject intercepts and fixed effects
m.cvfixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(m.cvfixed)

# BRMS model with random by-subject intercepts and fixed effects
bm.cvfixed = brm(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(bm.cvfixed)
stanplot(bm.cvfixed)

# compare coefficients of models with and without NN model predictions

bm.cvfixedandvectorpred.est = bm.cvfixedandvectorpred$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "fixed+nn predictions")

bm.cvfixed.est = bm.cvfixed$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "fixed")


estimates = rbind(bm.cvfixedandvectorpred.est, bm.ecvfixed.est)
estimates %>% ggplot(aes(y=str_replace(Parameter, "b_", ""), x=mu, col=model)) + 
  geom_point(size=3) + geom_point(size=2.0, col="white", alpha=0.6) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh), height=.1) +
  xlab("coefficient estimate") +
  ylab("Parameter") +
  theme(legend.position = "bottom")


# simulated estimates of P(beta_fixed < beta_nn) for all fixed effects

bm.cvfixed_df = bm.cvfixed$fit %>% 
  as.data.frame()

params = colnames(bm.cvfixed_df)[grepl("b_", colnames(bm.cvfixed_df))]

bm.cvfixed_df = bm.cvfixed_df %>%
  select(params)

bm.cvfixedandvectorpred_df = bm.cvfixedandvectorpred$fit %>%
  as.data.frame() %>%
  select(params)


n = 1000000
fixed_sample = bm.cvfixed_df %>% sample_n(size = n, replace = T)
nn_sample = bm.cvfixedandvectorpred_df %>% sample_n(size = n, replace = T)
sim_results = data.frame(fixed_sample < nn_sample)


sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise("P(beta_fixed < beta_nn)" = mean(value))

