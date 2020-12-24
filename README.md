# Using machine learning to analyze large scale qualitative data within seconds

Analysis of open-ended survey responses is hard work. It can take hours or even days to go through verbatim responses in a large survey dataset. Not only that, it is almost exclusively done through human coding. As a result, qualitative responses are often ignored or just used to supplement the narrative by pulling out a handful of verbatim quotes.

Which led me to the question - Is there a better way to reveal insights in open-ended survey responses?

Using a mixture of natural language processing, neural networks, sentiment analysis and topic modelling, I created a model that can take in a dataset, and automatically returns key themes in the data.

I used a publicly available dataset containing 3000 responses- from a community survey in Austin, Texas. At the end of the survey, respondents were given an option to provide written comments in response to the following question: “If there was ONE thing you could share with the Mayor regarding the City of Austin (any comment, suggestion, etc.), what would it be?” 

I thought this would be an interesting data science challenge to tackle given the wide range of possible responses.

Here's my machine learning pipeline -

# Step 1. Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is a popular Natural Language Processing (NLP) tool that can automatically identify topics from a corpus. LDA assumes each topic is made of a bag of words with certain probabilities, and each document is made of a bag of topics with certain probabilities. The goal of LDA is to learn the word and topic distribution underlying the corpus. Gensim is an NLP package that is particularly suited for LDA and other word-embedding machine learning algorithms, so I used this to implement my project.

After some preprocessing of the data to remove common words, I was able to obtain topics from respondent feedback that mainly revolved around -

1) Cost of living

2) Utilities

3) Traffic

4) Miscellaneous issues (classified as topic 0)

I also trained a word2vec neural network and projected the top words in the LDA-obtained general topics onto the word2vec space. We can visualise the clusters of topics in a 2D space using t-SNE (t-distributed stochastic neighbour embedding) algorithm. This allows us to see how the model has separated the 4 topics - Utilities issues appear to be the least commonly mentioned. Additionally, there is some overlap between cost of living and utilities issues.

And finally, I used pyLDAVis to create an interactive visualisation of this topic model. This can be used to chart the most salient words per topic, as well as seeing how separated the topics are.


# Step 2. Sentiment Analysis

I was also interested in analysing the sentiments of survey open-end comments, and merging that with my topic model. I used the VADER library to assign sentiment scores, and defined a percentage rating for a topic as the percent of respondents that gave a positive comment when they mentioned the topic. This metric was used to assign sentiment scores to topics.

I used my LDA model to determine the topic composition of each sentence in a response. If a sentence was dominated by one topic by 60% or more, I considered that sentence as belonging to that specific topic. Then, I calculated the sentiment of the sentence, either positive or negative, and finally counted the total percent of positive sentences in each topic.

# In conclusion, I am confident that this is an efficient and scalable way to analyse survey verbatim.

Some caveats - Since this is a 'quick and dirty' method, it cannot be expected to fully replace human analysis. Additionally, LDA also requires us to choose the number of topics which can be limiting. There is room for further improvement with more customised NLP depending on the subject matter, as well as better hyperparamter tuning in the LDA model. However, the upside is clear -

# Machine learning can help reduce human bias and save hours of analysis time in order to get major themes from your data. This particular method can easily handle large datasets and return actionable results within a matter of seconds.
