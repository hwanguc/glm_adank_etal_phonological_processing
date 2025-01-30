##########

## Author: Dr Han Wang
## This script contains data curation and analysis pipelines for Adank et al. (2025).


## Jan 30 2025 (V2): Added first-derivative analysis 
## Jun 6 2024 (V1): initial commitment.

###########


# Load packages and define some functions

library(car)
library(tidyverse)
library(dplyr)
library(tidyr)
library(reshape2)

library(ggplot2)
library(ggpubr)
library(ggsci)
library(ggeffects)
library(sjPlot)

library(forcats)

library(segmented)
library(lme4)
library(lmerTest)

library(mgcv)
library(mgcViz)
library(itsadug)

library(rcompanion)


`%notin%` <- Negate(`%in%`)

remove_outliers <- function(x, na.rm = TRUE, ...) {
  y <- x
  y[abs(x-mean(x)) > 3*sd(x)] <- NA
  y
}

sum_adjacent <- function(x) {
  if (length(x) < 2) {
    stop("Vector should have at least two elements.")
  }
  return(x[-length(x)] + x[-1])
}


# Data cleaning

## Speech task performance

dat_dual_speech_n192<-read.csv("dat_dysarthric_alltasks_replaced5.csv")
dat_dual_speech_n192$total_keyword<-as.numeric(dat_dual_speech_n192$total_keyword)
dat_dual_speech_n192$word_percent<-dat_dual_speech_n192$count_correct/dat_dual_speech_n192$total_keyword*100
dat_dual_speech_n192$word_proportion<-dat_dual_speech_n192$count_correct/dat_dual_speech_n192$total_keyword
dat_dual_speech_n192$trial<-as.numeric(dat_dual_speech_n192$trial)
dat_dual_speech_n192$participant<-as.factor(dat_dual_speech_n192$participant)
dat_dual_speech_n192$task<-as.factor(dat_dual_speech_n192$task)
dat_dual_speech_n192$sentence<-as.factor(dat_dual_speech_n192$sentence)
dat_dual_speech_n192$prompt<-as.factor(dat_dual_speech_n192$prompt)
dat_dual_speech_n192$correctness<-as.numeric(dat_dual_speech_n192$correctness)
dat_dual_speech_n192$rt<-as.numeric(dat_dual_speech_n192$rt)

library(Rmisc)
dat_dual_speech_n192_se <- summarySE(dat_dual_speech_n192, measurevar="word_percent", groupvars=c("trial","task")) # aggregate the raw data for plotting
dat_dual_speech_n192_se_perpar_pertask <-summarySE(dat_dual_speech_n192, measurevar="word_percent", groupvars=c("participant","task"))
dat_dual_speech_n192_se_pertask <-summarySE(dat_dual_speech_n192_se_perpar_pertask, measurevar="word_percent", groupvars=c("task"))
detach("package:Rmisc", unload=TRUE)


## Secondary task performance

dat_dual_n192<-dat_dual_speech_n192 %>% filter(task != 'speech_single')
dat_dual_n192_sndcorrect <- dat_dual_n192 %>% filter(correctness == 1)

library(Rmisc)
dat_dual_n192_secondary_acc <- summarySE(dat_dual_n192, measurevar="correctness", groupvars=c("trial","task"))
dat_dual_n192_secondary_acc_perpar_pertask <-summarySE(dat_dual_n192, measurevar="correctness", groupvars=c("participant","task"))
dat_dual_n192_secondary_acc_pertask <-summarySE(dat_dual_n192_secondary_acc_perpar_pertask, measurevar="correctness", groupvars=c("task"))


dat_dual_n192_sndcorrect_se<-summarySE(dat_dual_n192_sndcorrect, measurevar="rt", groupvars=c("trial","task"))
dat_dual_n192_sndcorrect_se_perpar_pertask <-summarySE(dat_dual_n192_sndcorrect, measurevar="rt", groupvars=c("participant","task"))
dat_dual_n192_sndcorrect_se_pertask <-summarySE(dat_dual_n192_sndcorrect_se_perpar_pertask, measurevar="rt", groupvars=c("task"))
detach("package:Rmisc", unload=TRUE)


## Effort and attention questionnaire


### Long format (for visualisation)

dat_question_n192_long<-read.csv("dat_questionnaire_long.csv") #Note that under task.type, a=single speech, b = dual-visual, c = dual-phonological, d = dual-lexical. Here, we removed the responses larger than 3SDs of the group mean and removed the NA rows for visual responses in speech single condition.

dat_question_n192_long_sdfiltered<-dat_question_n192_long %>%
  group_by(task.type,measure,task) %>%
  mutate_at(vars(rating), list(remove_outliers)) %>%
  as.data.frame()

### Wide format (for modelling)

dat_question_n192_wide_sdfiltered<-dcast(dat_question_n192_long_sdfiltered, task.type+participant ~ measure+task,value.var = "rating")
dat_question_n192_wide_sdfiltered_dual<-dat_question_n192_wide_sdfiltered[dat_question_n192_wide_sdfiltered$task.type!="a",]


# Figure 3: Trial-wise accuracy in speech task

## saturated model with a maximum random-effect structure that is allowed by the experimental design.

m_sp_log_glmer_full<-glmer(cbind(count_correct,total_keyword-count_correct)~1+log(trial)*task+(1+log(trial)|participant)+(1+task|sentence),
                           data=dat_dual_speech_n192, family = binomial(link = "logit"), 
                           control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=20e5))) # the model
summary(m_sp_log_glmer_full) # a summary output for the model's estimations



## best model

m_sp_log_glmer_2<-glmer(cbind(count_correct,total_keyword-count_correct)~1+log(trial)*task+(1+log(trial)|participant),
                        data=dat_dual_speech_n192, family = binomial(link = "logit"), 
                        control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=20e5)))
summary(m_sp_log_glmer_2)


m_sp_log_glmer_2_phon<-glmer(cbind(count_correct,total_keyword-count_correct)~1+log(trial)*relevel(as.factor(task), ref="phonological")+(1+log(trial)|participant),
                             data=dat_dual_speech_n192, family = binomial(link = "logit"), 
                             control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=20e5))) # change the reference level to phonological task.
summary(m_sp_log_glmer_2_phon)


m_sp_log_glmer_2_vis<-glmer(cbind(count_correct,total_keyword-count_correct)~1+log(trial)*relevel(as.factor(task), ref="visual")+(1+log(trial)|participant),
                            data=dat_dual_speech_n192, family = binomial(link = "logit"), 
                            control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=20e5)))
summary(m_sp_log_glmer_2_vis)

m_sp_log_glmer_2_sin<-glmer(cbind(count_correct,total_keyword-count_correct)~1+log(trial)*relevel(as.factor(task), ref="speech_single")+(1+log(trial)|participant),
                            data=dat_dual_speech_n192, family = binomial(link = "logit"), 
                            control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=20e5)))
summary(m_sp_log_glmer_2_sin)


gg_m_sp_log_glmer_2<-ggpredict(m_sp_log_glmer_2, term = c("trial[all]","task"))
plot(gg_m_sp_log_glmer_2)

gg_m_sp_log_glmer_2<-data.frame(gg_m_sp_log_glmer_2)
gg_m_sp_log_glmer_2$task<-gg_m_sp_log_glmer_2$group

## refitted model in `afex` for pair-wise comparison.

library(afex)
library(emmeans)
m_sp_2_afex<-(mixed(cbind(count_correct,total_keyword-count_correct)~1+log(trial)*task+(1+log(trial)|participant),
                    data=dat_dual_speech_n192,method = "LRT",family = binomial(link = "logit"),
                    control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5))))
summary(m_sp_2_afex)

(emm_m_sp_2_afex<-emmeans(m_sp_2_afex,"task")) # estimation for task-wise performance using library(afex)
pairs(emm_m_sp_2_afex,adjust="holm") # pair-wise comparison results for task conditions

detach("package:afex", unload=TRUE)

anova(m_sp_log_glmer_full,m_sp_log_glmer_2)



### plot

m_sp_2_reorder<-c('speech_single','visual','phonological','lexical') # reorder the levels for plotting

facet_labels<-c(lexical = "Dual Lexical", phonological = "Dual Phonological",speech_single = 'Single', visual = 'Dual Visual') # Set the label for panels in the plot

dat_dual_speech_n192_se_reorder <- dat_dual_speech_n192_se %>% 
  mutate(task = fct_relevel(task, m_sp_2_reorder)) # reorder the raw data per level of task

m_sp_2_prediction_reorder<-gg_m_sp_log_glmer_2 %>% 
  mutate(task = fct_relevel(task, m_sp_2_reorder)) # reorder the prediction per level of task

plot_m_sp_2<-ggplot(dat_dual_speech_n192_se_reorder,aes(x=trial, y=word_percent,color=task, shape=task)) + 
  geom_errorbar(aes(ymin=word_percent-se, ymax=word_percent+se), width=.1) +
  geom_point() +
  geom_line(data=m_sp_2_prediction_reorder, aes(x=x, y=predicted*100,color=task), size=.6) +
  geom_ribbon(data=m_sp_2_prediction_reorder, aes(ymin=conf.low*100, ymax=conf.high*100, x=x, y=predicted*100,fill=task,color=task), alpha = 0.2) +# error band
  scale_y_continuous(breaks = seq(40, 100, by = 10),limits=c(40, 100))+
  labs(x="Trial number", y = "%Correct")+
  theme_minimal()+
  facet_wrap(~task,
             labeller = labeller(task = facet_labels)) # main figure


plot_m_sp_2+
  theme(plot.title = element_text(size = 13,hjust=0.5), 
        axis.text.x = element_text(size=11.5,angle = 0, hjust = 0.5),
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        strip.text.x = element_text(size = 11),
        legend.title = element_text(size=12),
        legend.text = element_text(size=11),
        legend.position = "none") # update the formatting for the plot



# Figure 4: Trial-wise first-derivative in speech task

gg_m_sp_log_glmer_2<-gg_m_sp_log_glmer_2 %>%
  group_by(task) %>%
  mutate(first_derivatives = c(NA,diff(predicted)/diff(x)))


plot_m_sp_2_first_dev_noci<-ggplot(m_sp_2_prediction_reorder,aes(x=x, y=first_derivatives,color=task)) + 
  geom_line(size=.6) +
  labs(x="Trial number", y = "First derivatives")+
  theme_minimal()+
  facet_wrap(~task,
             labeller = labeller(task = facet_labels)) # main figure


plot_m_sp_2_first_dev_noci+
  theme(plot.title = element_text(size = 13,hjust=0.5), 
        axis.text.x = element_text(size=11.5,angle = 0, hjust = 0.5),
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        strip.text.x = element_text(size = 11),
        legend.title = element_text(size=12),
        legend.text = element_text(size=11),
        legend.position = "none") # update the formatting for the plot


# Figure 5: Trial-wise accuracy in secondary tasks

## saturated model: see comments on Figure 3 for how the code is structured

m_2nd_log_glmer_full<-glmer(correctness~1+log(trial)*task+(1+log(trial)|participant)+(1+task|prompt),
                            data=dat_dual_n192, family = binomial(link = "logit"), 
                            control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))
summary(m_2nd_log_glmer_full)


## glmer_2: best model

m_2nd_log_glmer_2<-glmer(correctness~1+log(trial)*task+(1+log(trial)|participant)+(1|prompt),
                         data=dat_dual_n192, family = binomial(link = "logit"), 
                         control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))
summary(m_2nd_log_glmer_2)


m_2nd_log_glmer_2_phon<-glmer(correctness~1+log(trial)*relevel(as.factor(task), ref="phonological")+(1+log(trial)|participant)+(1|prompt),
                              data=dat_dual_n192, family = binomial(link = "logit"), 
                              control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))
summary(m_2nd_log_glmer_2_phon)



gg_m_2nd_log_glmer_2<-ggpredict(m_2nd_log_glmer_2, terms = c("trial[all]","task")) # save the prediction from the model using ggpredict()
plot(gg_m_2nd_log_glmer_2) # plot the model

gg_m_2nd_log_glmer_2$task<-gg_m_2nd_log_glmer_2$group


### plot

m_2nd_2_reorder<-c('visual','phonological','lexical') # reorder the levels for plotting

facet_labels<-c(lexical = "Dual Lexical",phonological = 'Dual Phonological', visual = 'Dual Visual') # Set the label for panels in the plot

dat_dual_n192_secondary_acc_reorder <- dat_dual_n192_secondary_acc %>% 
  mutate(task = fct_relevel(task, m_2nd_2_reorder)) # reorder the raw data per level of task

m_2nd_2_prediction_reorder<-gg_m_2nd_log_glmer_2 %>% 
  mutate(task = fct_relevel(task, m_2nd_2_reorder)) # reorder the prediction per level of task

plot_m_2nd_2<-ggplot(dat_dual_n192_secondary_acc_reorder,aes(x=trial, y=correctness*100,color=task, shape=task)) + 
  geom_errorbar(aes(ymin=(correctness-se)*100, ymax=(correctness+se)*100), width=.1) +
  geom_point() +
  geom_line(data=m_2nd_2_prediction_reorder, aes(x=x, y=predicted*100,color=task), size=.6) +
  geom_ribbon(data=m_2nd_2_prediction_reorder, aes(ymin=conf.low*100, ymax=conf.high*100, x=x, y=predicted*100,fill=task,color=task), alpha = 0.2) +# error band
  scale_y_continuous(breaks = seq(40, 100, by = 10),limits=c(40, 100))+
  labs(x="Trial number", y = "%Correct")+
  theme_minimal()+
  facet_wrap(~task,
             labeller = labeller(task = facet_labels)) # main figure


plot_m_2nd_2+
  theme(plot.title = element_text(size = 13,hjust=0.5), 
        axis.text.x = element_text(size=11.5,angle = 0, hjust = 0.5),
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        strip.text.x = element_text(size = 11),
        legend.title = element_text(size=12),
        legend.text = element_text(size=11),
        legend.position = "none") # update the formatting for the plot

anova(m_2nd_log_glmer_2,m_2nd_log_glmer_full)


library(afex)
library(emmeans)
m_2nd_2_afex<-(mixed(correctness~1+log(trial)*task+(1+log(trial)|participant)+(1|prompt),
                     data=dat_dual_n192,method = "LRT",family = binomial(link = "logit"),
                     control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5))))
summary(m_2nd_2_afex)
(emm_m_2nd_2_afex<-emmeans(m_2nd_2_afex,"task"))
pairs(emm_m_2nd_2_afex,adjust="holm")
detach("package:afex", unload=TRUE)


# Figure 6: Trial-wise RTs in secondary tasks

## saturated model: see comments on Figure 3 for how the code is structured

m_rt_glmer_full<-glmer(rt~1+log(trial)*task+(1+log(trial)|participant)+(1+task|prompt),
                       data=dat_dual_n192_sndcorrect, family = Gamma(link = "log"), 
                       control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))

summary(m_rt_glmer_full)


## glmer_3: best-fitting model

m_rt_glmer_3<-glmer(rt~1+log(trial)*task+(1+task|prompt),
                    data=dat_dual_n192_sndcorrect, family = Gamma(link = "log"), 
                    control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))

summary(m_rt_glmer_3)


m_rt_glmer_3_phon<-glmer(rt~1+log(trial)*relevel(as.factor(task), ref="phonological")+(1+task|prompt),
                         data=dat_dual_n192_sndcorrect, family = Gamma(link = "log"), 
                         control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))

summary(m_rt_glmer_3_phon)


library(afex)
library(emmeans)
m_rt_glmer_3_afex<-(mixed(rt~1+log(trial)*task+(1+task|prompt),
                          data=dat_dual_n192_sndcorrect,method = "LRT",family = Gamma(link = "log"),
                          control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5))))
summary(m_rt_glmer_3_afex)
(emm_m_rt_glmer_3_afex<-emmeans(m_rt_glmer_3_afex,"task"))
pairs(emm_m_rt_glmer_3_afex,adjust="holm")
detach("package:afex", unload=TRUE)


gg_m_rt_glmer_3<-ggpredict(m_rt_glmer_3, terms = c("trial[all]","task")) # save the prediction from the model using ggpredict()
plot(gg_m_rt_glmer_3) # plot the model

gg_m_rt_glmer_3<-data.frame(gg_m_rt_glmer_3)
gg_m_rt_glmer_3$task<-gg_m_rt_glmer_3$group


### plot

m_rt_final_reorder<-c('visual','phonological','lexical') # reorder the levels for plotting

facet_labels<-c(lexical = "Dual Lexical",phonological = 'Dual Phonological', visual = 'Dual Visual') # Set the label for panels in the plot

dat_dual_n192_sndcorrect_se_reorder <- dat_dual_n192_sndcorrect_se %>% 
  mutate(task = fct_relevel(task, m_rt_final_reorder)) # reorder the raw data per level of task

m_rt_final_prediction_reorder<-gg_m_rt_glmer_3 %>% 
  mutate(task = fct_relevel(task, m_rt_final_reorder)) # reorder the prediction per level of task

plot_m_rt_final<-ggplot(dat_dual_n192_sndcorrect_se_reorder,aes(x=trial, y=rt,color=task, shape=task)) + 
  geom_errorbar(aes(ymin=rt-se, ymax=rt+se), width=.1) +
  geom_point() +
  geom_line(data=m_rt_final_prediction_reorder, aes(x=x, y=predicted,color=task), size=.6) +
  geom_ribbon(data=m_rt_final_prediction_reorder, aes(ymin=conf.low, ymax=conf.high, x=x, y=predicted,fill=task,color=task), alpha = 0.2) +# error band
  scale_y_continuous(breaks = seq(400, 1400, by = 200),limits=c(400, 1400))+
  labs(x="Trial number", y = "RT (ms)")+
  theme_minimal()+
  facet_wrap(~task,
             labeller = labeller(task = facet_labels)) # main figure


plot_m_rt_final+
  theme(plot.title = element_text(size = 13,hjust=0.5), 
        axis.text.x = element_text(size=11.5,angle = 0, hjust = 0.5),
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        strip.text.x = element_text(size = 11),
        legend.title = element_text(size=12),
        legend.text = element_text(size=11),
        legend.position = "none") # update the formatting for the plot



# Figure C1: Effort and attention questionnaire

## Figure

dat_question_n192_long_sdfiltered$measure <- factor(dat_question_n192_long_sdfiltered$measure, levels = c("effort","attention")) # factorise the data
dat_question_n192_long_sdfiltered$task <- factor(dat_question_n192_long_sdfiltered$task, levels = c("speech","concurrent"))

facet_labels<-c(speech = "Speech task response", concurrent = "Concurrent task response") # set labels for plotting

plot_effortques_speechvisual_sdfiltered<-ggplot(dat_question_n192_long_sdfiltered, aes(x=measure, y=rating, fill=task.type)) +
  geom_boxplot(outlier.shape=NA,position=position_dodge(width=0.9))+
  stat_summary(fun.y = "mean", geom = "point", aes(group=task.type), shape = 23, size = 3, fill = "grey",position=position_dodge(0.9)) +
  geom_point(aes(group=task.type),size=2,shape=21, position = position_jitterdodge(dodge.width = 0.9, jitter.width = 0.15), alpha = 0.55)+
  #geom_jitter()+
  scale_x_discrete(labels=c("effort" = "Effort", attention = "Attention"))+
  scale_y_continuous(breaks = seq(0, 100, by = 10))+
  scale_fill_npg(labels = c("Single", "Visual","Phonological","Lexical"))+
  #scale_colour_npg(labels = c("Single", "Visual","Phonological","Lexical"))+
  labs(x="Measure", y = "Participant estimate",
       fill = "Task")+
  facet_wrap(~task,labeller = labeller(task = facet_labels))+
  theme_minimal()


plot_effortques_speechvisual_sdfiltered+
  theme(plot.title = element_text(size = 13,hjust=0.5), 
        axis.text.x = element_text(size=11.5,angle = 0, hjust = 0.5),
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        strip.text.x = element_text(size = 11.5),
        legend.title = element_text(size=12),
        legend.text = element_text(size=11))

## Models:

#### effort speech (no outlier)

ques_speechvisual_speecheffort_m_lm_nooutlier<-lm(effort_speech ~ relevel(as.factor(task.type), ref="a"), na.action = na.omit, data=dat_question_n192_wide_sdfiltered)

summary(ques_speechvisual_speecheffort_m_lm_nooutlier) # Model output can be found in Table B6


#### attention speech (no outlier)

ques_speechvisual_speechattention_m_lm_nooutlier<-lm(attention_speech ~ relevel(as.factor(task.type), ref="a"), na.action = na.omit, data=dat_question_n192_wide_sdfiltered)

summary(ques_speechvisual_speechattention_m_lm_nooutlier) # Model output can be found in Table B6


#### effort concurrent (no outlier)

ques_speechtsecondary_tsecondaryeffort_m_lm_nooutlier<-lm(effort_concurrent ~ relevel(as.factor(task.type), ref="b"), na.action = na.omit, data=dat_question_n192_wide_sdfiltered_dual)

summary(ques_speechtsecondary_tsecondaryeffort_m_lm_nooutlier) # Model output can be found in Table B6


#### attention secondary (no outlier)

ques_speechtsecondary_tsecondaryattention_m_lm_nooutlier<-lm(attention_concurrent ~ relevel(as.factor(task.type), ref="b"), na.action=na.omit, data=dat_question_n192_wide_sdfiltered_dual)

summary(ques_speechtsecondary_tsecondaryattention_m_lm_nooutlier) # Model output can be found in Table B6


# Figure C2: correlations between individual slopes of the speech and secondary task conditions

## data curation

dat_randslopes<-dat_dual_speech_n192 %>%
  filter(task != "speech_single" & trial <= 20)

## extract random slopes per condition from the models and store them to a spreadsheet

coefs<-data.frame(matrix(,nrow=48,ncol=6))

tasks<-c("sp","2nd")
conds<-c("visual","phonological","lexical")

i<-1

# extract the random slopes

for (cond in conds) {
  
  # fit a model
  
  current_m_sp<-glmer(cbind(count_correct,total_keyword-count_correct)~1+log(trial)*relevel(as.factor(task), ref=cond)+(1+log(trial)|participant),
                      data=dat_randslopes, family = binomial(link = "logit"), 
                      control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))
  current_m_2nd<-glmer(correctness~1+log(trial)*relevel(as.factor(task), ref=cond)+(1+log(trial)|participant),
                       data=dat_randslopes, family = binomial(link = "logit"), 
                       control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=30e5)))
  
  current_m_sp_coef<-coef(current_m_sp)$participant
  current_m_2nd_coef<-coef(current_m_2nd)$participant
  
  dat_currentcond_subj_lst<-dat_randslopes %>%
    filter(task == cond) %>%
    pull(participant) %>%
    as.character() %>%
    unique()
  
  dat_curretcond_subj_idx<-is.element(row.names(current_m_sp_coef),dat_currentcond_subj_lst)
  
  m_sp_coef<-current_m_sp_coef[dat_curretcond_subj_idx,]
  
  names(coefs)[i]<-paste("logtrial",tasks[1],cond,sep = "_")
  coefs[i]<-m_sp_coef$`log(trial)`
  i<-i+1
  
  
  m_2nd_coef<-current_m_2nd_coef[dat_curretcond_subj_idx,]
  
  names(coefs)[i]<-paste("logtrial",tasks[2],cond,sep = "_")
  coefs[i]<-m_2nd_coef$`log(trial)`
  i<-i+1
  
} 

## Plot

sp_vis<-ggplot(data = coefs, aes(x=logtrial_sp_visual, y=logtrial_2nd_visual)) + 
  geom_point()+
  geom_smooth(method=lm)+
  stat_cor(p.accuracy = 0.001, r.accuracy = 0.01)+
  labs(x="Speech task under visual task", y = "Visual task")+
  theme_bw()

sp_vis_formatted<-sp_vis+theme(axis.title.x = element_text(size=14,angle = 0, hjust = 0.5),
                               axis.title.y = element_text(size=14,angle = 90, hjust = 0.5),
                               axis.text.x = element_text(size=12,angle = 0, hjust = 0.5),
                               axis.text.y = element_text(size=12,angle = 0, hjust = 0.5),
                               panel.grid.major.x = element_line(color = "grey90"),
                               panel.grid.minor.x = element_line(color = "grey90"))


sp_phon<-ggplot(data = coefs, aes(x=logtrial_sp_phonological, y=logtrial_2nd_phonological)) + 
  geom_point()+
  geom_smooth(method=lm)+
  stat_cor(p.accuracy = 0.001, r.accuracy = 0.01)+
  labs(x="Speech task under phonological task", y = "Phonological task")+
  theme_bw()

sp_phon_formatted<-sp_phon+theme(axis.title.x = element_text(size=14,angle = 0, hjust = 0.5),
                                 axis.title.y = element_text(size=14,angle = 90, hjust = 0.5),
                                 axis.text.x = element_text(size=12,angle = 0, hjust = 0.5),
                                 axis.text.y = element_text(size=12,angle = 0, hjust = 0.5),
                                 panel.grid.major.x = element_line(color = "grey90"),
                                 panel.grid.minor.x = element_line(color = "grey90"))



sp_lex<-ggplot(data = coefs, aes(x=logtrial_sp_lexical, y=logtrial_2nd_lexical)) + 
  geom_point()+
  geom_smooth(method=lm)+
  stat_cor(p.accuracy = 0.001, r.accuracy = 0.01)+
  labs(x="Speech task under lexical task", y = "Lexical task")+
  theme_bw()

sp_lex_formatted<-sp_lex+theme(axis.title.x = element_text(size=14,angle = 0, hjust = 0.5),
                               axis.title.y = element_text(size=14,angle = 90, hjust = 0.5),
                               axis.text.x = element_text(size=12,angle = 0, hjust = 0.5),
                               axis.text.y = element_text(size=12,angle = 0, hjust = 0.5),
                               panel.grid.major.x = element_line(color = "grey90"),
                               panel.grid.minor.x = element_line(color = "grey90"))


task_all <- ggarrange(sp_vis_formatted,sp_phon_formatted,sp_lex_formatted,
                      font.label = list(size = 12, face = "bold", color ="black"),ncol = 3, nrow = 1)


annotate_figure(task_all,
                bottom = text_grob("Beta estimate (Speech task)",
                                   hjust = 0.5, size = 15),
                left = text_grob("Beta estimate (Concurrent task)", rot = 90,size = 15)
)

