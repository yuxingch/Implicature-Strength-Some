library(tidyverse)
library(brms)
library(lme4)
library(lmerTest)
library(stringr)


#setwd("analysis/rscripts/")

setwd("~/Dropbox/Uni/RA/implicature-strength/Research/analysis/rscripts/")
theme_set(theme_bw())

# color-blind-friendly palette
cbPalette <- c("#0072B2",  "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00") 

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
bm.fixed = brm(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=centered, control = list(max_treedepth = 12))
summary(bm.fixed)


################################################
#  Models estimated only from evaluation data  #
################################################


# Add Yuxing's predictions
preds = read_tsv("../../runs.eval/bert_large_lstm_eval/Preds/test_preds_rating_epoch190.csv") %>%
  rename(Item = Item_ID)

d = centered %>%
  right_join(d_no_context) %>%
  right_join(preds)

# LMER model with random by-subject intercepts and model predictions 
m.evalvectorpred = lmer(Rating ~ predicted + (1|workerid), data=d)
summary(m.evalvectorpred) 

# BRMS model with random by-subject intercepts and model predictions 
bm.evalvectorpred = brm(Rating ~ predicted + (1|workerid), data=d)
summary(bm.evalvectorpred) 

# LMER model with random by-subject intercepts, fixed effects, and model predictions 
m.evalfixedandvectorpred = lmer(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid) + (1|Item), data=d)
summary(m.evalfixedandvectorpred)

# BRMS model with random by-subject intercepts, fixed effects, and model predictions 
bm.evalfixedandvectorpred = brm(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid) + (1|Item), data=d, control = list(max_treedepth = 12))
summary(bm.evalfixedandvectorpred)
stanplot(bm.evalfixedandvectorpred)

# LMER model with random by-subject intercepts, and fixed effects
m.evalfixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(m.evalfixed)

# BRMS model with random by-subject intercepts,and fixed effects
bm.evalfixed = brm(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d, control = list(max_treedepth = 12))
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
  mutate(model = "extended model")

bm.evalfixed.est = bm.evalfixed$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "original model")


factor_levels = c("Intercept", "cPartitive", 
                  "cStrengthSome", "credInfoStatus", 
                  "cBinaryGF", "cModification",
                  "clogSentenceLength", "cPartitive:cStrengthSome",
                  "credInfoStatus:cBinaryGF", "credInfoStatus:cModification",
                  "cBinaryGF:cModification", "credInfoStatus:cBinaryGF:cModification", 
                  "predicted")
factor_labels = c("Intercept", "Partitive", "Strength", "Linguistic mention", "Subjecthood", "Modification", "Utterance length", "Partitive:Strength",
                  "Linguistic mention:Subjecthood", "Linguistic mention:Modification", "Subjecthood:Modification", "Linguistic mention:Subjecthood:\nModification", "NN prediction")
factor_levels = rev(factor_levels)
factor_labels = rev(factor_labels)


estimates = rbind(bm.evalfixedandvectorpred.est, bm.evalfixed.est)
estimates$Parameter = str_replace(estimates$Parameter, "b_", "")
estimates = estimates %>% filter(Parameter != "Intercept")
estimates$Parameter = factor(estimates$Parameter, levels = factor_levels, labels=factor_labels, ordered = TRUE)
estimates$model = factor(estimates$model, levels = c("original model", "extended model"), ordered = TRUE)
estimates %>% ggplot(aes(y=str_replace(Parameter, "b_", ""), x=mu, col=model)) + 
  geom_point(size=3) + geom_point(size=2.0, col="white", alpha=0.6) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh), height=.1) +
  xlab("coefficient estimate") +
  ylab("Parameter") +
  theme(legend.position = "bottom") +
  scale_color_manual(values=cbPalette)

max_values = estimates %>% group_by(Parameter) %>% summarise(mean_estimate = max(cihigh))

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
sim_results = data.frame(abs(fixed_sample) < abs(nn_sample))


sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise("P(beta_fixed < beta_nn)" = mean(value))

signif_values = sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise(p = mean(value)) %>%
  mutate(label=cut(p, breaks=c(-.01,0.001,0.01,0.05,1.0),
                   labels=c("***", "**", "*", ""))) %>%
  mutate(Parameter = factor(str_replace(str_replace_all(Parameter, "\\.", ":"), "b_", ""), levels = factor_levels, labels=factor_labels)) %>%
  merge(max_values)

estimates_plot = estimates %>% ggplot(aes(y=Parameter, x=mu, col=model)) + 
  geom_vline(xintercept=0) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh), size=1, height=.4) +
  geom_point(size=5) + geom_point(size=3.0, col="white", alpha=0.6) +
  xlab("Coefficient estimate") +
  ylab("Parameter") +
  geom_text(aes(x=mean_estimate, label=label),col="black", size=5, data=signif_values, nudge_x=.1, nudge_y=-.1) + 
  theme(legend.position = "bottom")  +
  guides(color=guide_legend(title="Regression model"))

ggsave(estimates_plot, filename = "../graphs/eval_coefficient_estimates.pdf", width=24, height=12, units = "cm")


#visualization of correlation

r2e = geom_text(x=1, y=6.5, hjust=0, label=as.character(as.expression(" "~R^2~"= 0.776")), color="black", parse=TRUE, size=5)


eval_correlation_plot = d %>%
  group_by(Item) %>%
  summarise(predicted = mean(predicted), original_mean = mean(original_mean)) %>%
  ggplot(aes(y=original_mean, x=predicted)) + 
  geom_point() + 
  geom_smooth(method="lm") +
  ylab("Empirical ratings") +
  xlab("Model predictions") +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7), limits=c(1,7)) +
  scale_y_continuous(breaks=c(1,2,3,4,5,6,7), limits=c(1,7)) +
  theme(axis.title = element_text(size=14)) + 
  r2e

ggsave(eval_correlation_plot, filename = "../graphs/model_fit_eval.pdf", width=12, height=10, units = "cm")

cor_data = d %>%
  group_by(Item) %>%
  summarise(predicted = mean(predicted), original_mean = mean(original_mean))
  
cor(cor_data$original_mean, cor_data$predicted)

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

d_no_context = read_csv("../data/some_without_context_means.csv")   %>%
  rename(Item = tgrep_id, original_mean_nocontext = Mean) %>%
  select(Item, original_mean_nocontext)
  

d = centered %>%
  right_join(preds) %>%
  right_join(d_no_context)



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
bm.cvfixedandvectorpred = brm(Rating ~ predicted + cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d, control = list(max_treedepth = 12))
summary(bm.cvfixedandvectorpred)
stanplot(bm.cvfixedandvectorpred)

# LMER model with random by-subject intercepts, and fixed effects
m.cvfixed = lmer(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d)
summary(m.cvfixed)

# BRMS model with random by-subject intercepts, and fixed effects
bm.cvfixed = brm(Rating ~ cPartitive*cStrengthSome+credInfoStatus*cBinaryGF*cModification + clogSentenceLength + (1|workerid), data=d, control = list(max_treedepth = 12))
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
  mutate(model = "extended model")

bm.cvfixed.est = bm.cvfixed$fit %>% 
  as.data.frame() %>% 
  gather(key = "Parameter") %>% 
  filter(grepl("b_", Parameter)) %>% 
  group_by(Parameter) %>% 
  summarise(mu = mean(value), 
            cilow = quantile(value, 0.025), 
            cihigh = quantile(value, 0.975)) %>%
  mutate(model = "original model")


estimates = rbind(bm.cvfixedandvectorpred.est, bm.cvfixed.est)
estimates$Parameter = str_replace(estimates$Parameter, "b_", "")
estimates = estimates %>% filter(Parameter != "Intercept")
estimates$Parameter = factor(estimates$Parameter, levels = factor_levels, labels=factor_labels, ordered = TRUE)
estimates$model = factor(estimates$model, levels = c("original model", "extended model"), ordered = TRUE)

max_values = estimates %>% group_by(Parameter) %>% summarise(mean_estimate = max(cihigh))




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
sim_results = data.frame(abs(fixed_sample) < abs(nn_sample))


signif_values = sim_results %>% 
  gather(key="Parameter") %>% 
  group_by(Parameter) %>% 
  summarise(p = mean(value)) %>%
  mutate(label=cut(p, breaks=c(-.01,0.001,0.01,0.05,1.0),
                   labels=c("***", "**", "*", ""))) %>%
  mutate(Parameter = factor(str_replace(str_replace_all(Parameter, "\\.", ":"), "b_", ""), levels = factor_levels, labels=factor_labels)) %>%
  merge(max_values)

estimates_plot = estimates %>% ggplot(aes(y=Parameter, x=mu, col=model)) + 
  geom_vline(xintercept=0) +
  geom_errorbarh(aes(xmin=cilow, xmax=cihigh), size=1, height=.4) +
  geom_point(size=5) + geom_point(size=3.0, col="white", alpha=0.6) +
  xlab("Coefficient estimate") +
  ylab("Parameter") +
  geom_text(aes(x=mean_estimate, label=label),col="black", size=5, data=signif_values, nudge_x=.1, nudge_y=-.1) + 
  theme(legend.position = "bottom")  +
  guides(color=guide_legend(title="Regression model"))

ggsave(estimates_plot, filename = "../graphs/cv_coefficient_estimates.pdf", width=24, height=12, units = "cm")

######################################
# analysis of no-context data
######################################


#item_means = d %>%
#  arrange(original_mean) %>%
#  mutate(pos=seq(1,nrow(d))) %>%
#  gather(key="source", value="rat", original_mean, original_mean_nocontext, predicted) %>% 
#  group_by(Item, source) %>% 
#  summarise(rat=mean(rat), pos = mean(pos)) %>%
#  arrange(pos)
#
#item_means$sorted_idx = seq.int(nrow(item_means))
#item_means$g = round(item_means$sorted_idx / (nrow(item_means) / 6))
#item_means$Item = factor(item_means$Item, levels=unique(item_means$Item))
#item_means %>%  ggplot(aes(x=Item, y=rat, col=source)) + geom_point() + facet_wrap(~g, scales = "free_x", nrow = 7) 
#
#
#d %>%
#  group_by(Item) %>%
#  summarise(predicted = mean(predicted), original_mean = mean(original_mean), original_mean_nocontext = mean(original_mean_nocontext)) %>%
#  ggplot(aes(x=original_mean, y=predicted)) + geom_point() + geom_smooth(method="lm")
#
#d %>%
#  group_by(Item) %>%
#  summarise(predicted = mean(predicted), original_mean = mean(original_mean), original_mean_nocontext = mean(original_mean_nocontext)) %>%
#  ggplot(aes(x=original_mean_nocontext, y=predicted)) + geom_point() + geom_smooth(method="lm")
#
#d %>%
#  group_by(Item) %>%
#  summarise(predicted = mean(predicted), original_mean = mean(original_mean), original_mean_nocontext = mean(original_mean_nocontext)) %>%
#  mutate(diff_predicted_context = abs(predicted-original_mean)) %>%
#  mutate(diff_predicted_nocontext = abs(predicted-original_mean_nocontext)) %>%
#  mutate(diff_context_nocontext = abs(original_mean-original_mean_nocontext)) %>%
#  arrange(desc(diff_predicted_context))
#  
#  
#
#cor(d$predicted, d$original_mean)
#
#cor(d$predicted, d$original_mean_nocontext)
#
#cor(d$original_mean, d$original_mean_nocontext)
#
#
#