# Sentiment Analysis on International News
## Abstract
In our study, we're diving into the world of newspapers to understand how they express different opinions on various topics through sentiment analysis. First off, we explain how we gathered a bunch of data from the web using specific methods. We talk about the data we collected and then explain the steps we took to get it ready for analysis. This includes counting words, creating bags of words, and using TF-IDF to understand how important certain words are.
Once our data is prepared, we move on to the analysis. We use word clouds to show which words appear most frequently, and we use clustering methods to see if there are connections between five different news editors we chose. This helps us figure out how they report and what opinions they might hold.
After that, we check how well Python and Spark perform in handling the data collected, with the main purpose being to explore which one would be more efficient in this type of task.
Moving on, we explore the emotions expressed in the news articles using a pre-trained model for sentiment analysis and visualize using ggplot2. This helps us understand the overall feelings conveyed in the news.
In the final part, we try something different. We attempt to make the news articles sound more neutral or unbiased. To do this, we use ChatGPT 3.5. This experiment shows us if advanced language tools can actually influence the language used in news articles to sound more neutral. It's all about understanding how technology can impact the way news is written and the tones it carries.
## Introduction and Motivation
Online news is an important source of information nowadays. People rely on news to understand what is happening in the world. This is because people expect and trust news to provide the most reliable information to its audience. However, the neutrality of news has become a problem. There are increasing criticisms related to how news is becoming more biased and emotional. 
In this project, we decided to analyze the tendency of different news worldwide towards various topics. The result of this project can warn the audience to keep alerted when seeing the media reporting related news.
## Objective
To find reliable news organizations that present reliable information, we decided to apply sentiment analysis to online news in different countries and regions. We aim to provide a clearer view of the credibility of the most popular media around the world. Moreover, we try to figure out subtle tendencies of the news toward certain topics, to warn the readers from being misled by the report.
Additionally, we want to find out whether ChatGPT can do better than the reporters. Consequently, we fed the scraped text data into ChatGPT and asked the bot to make the sentence more neutral and objective.
## Data Sources
We chose CNN from the US, China Daily from Mainland China, SCMP from Hong Kong, and The Sun and The Guardian from the UK. The reason why we chose two media from the UK is because The Sun is famous for its vulgar and slanted content while The Guardian has a good reputation.
In order to magnify the differences between them, we selected five controversial keywords: ChatGPT, Elon Musk, Donald Trump, Israel, and LGBT. These topics are all recent, for example, the war between Israel and Hamas just got started a few months ago.
## Methodology and Discussion
### Web Scraping
First of all, we copied the URLs of all the news websites. We tried to use Octoparse introduced in the tutorial, but the websites have unique ways to block the scrapers. Thus, we wrote custom Python code for each website. We tried a library called Requests, but the JavaScript code of the websites will block the access of the scraper. Finally, we use another library, Selenium, to control a browser so that the website cannot identify whether this is a scraper or a human user.
After we found a possible way to collect the data, we needed to find the URL of the news. We identified the change in the URLs when we searched for those keywords. For example, the main page of SCMP is “https://www.scmp.com/,” and after searching the keyword “ChatGPT” on the website, the URL became “https://www.scmp.com/search/ChatGPT.” Then, we identified all the news links that appeared under the search bar. We collected all the URLs and scraped the text content inside each link.
We faced many challenges in the web scraping process. All the websites have their own way of preventing web crawlers. One common way is that the website will delay for a certain period of time before showing the texts. To tackle this problem, we also set a delay when reading the content. But if the delay was set too long, the website might hide the content again and ask the user to pay to see more. 
As a result, the time interval should be set precisely. Some news websites do not provide enough text content. For instance, the BBC only provides news videos with a little description, so we changed to The Guardian afterward.
In total, we collected five media with five keywords and twenty reports for each keyword. There are roughly ten sentences in a report, this gives us  5*5*20*10=5000 sentences. We could scrape more data from some of the websites, but that would make the data unbalanced among the media because some websites are much harder to crawl than others.
### Data Preprocessing
Firstly, we have cleaned the data scraped from the news, such as removing HTML tags, special characters, and punctuation marks. After performing the data cleaning, we further process the data in three main techniques, consisting of word count, bag of words, and TF-IDF, to extract meaningful information and prepare the data for our further analysis, such as clustering and sentiment analysis.
As we are also using a pre-trained model to perform sentiment analysis using Python, besides the three techniques described before, we have to prepare the data by applying lemmatization, tokenization, and stemming. We use the Python Library NLTK to preprocess the text. As learned in the course lecture, the three are useful techniques to transform the original data. Word tokenization breaks the sentences into smaller units containing individual words. The word lemmatization will turn the word into its base form without destroying the meaning of the text. Stopwords are also removed from the text to reduce redundant processing time. Word stemming on the other hand reduces the word to its root form.
### Sentiment Prediction
To get the sentiment from the preprocessed text data, we use a pre-trained model to predict the sentiment. It outputs a score from 0, which means negative, to 1, which means positive. The pre-trained model is trained on the Sentiment140 dataset with 1.6 million tweets [1]. By using a pre-trained model to predict the sentiment, a quantitative score representing the sentiment of the text can be output for a further analysis.
We use Keras library in Python to load, and run the prediction on the text.
Finally, the original unstructured text is transformed and recorded in structured CSV format by the Panda library for other team members to quickly process.
### Word Count
The word count approach is probably the most straightforward technique to analyze the scraped data, but nonetheless, it offers several advantages. The first one is its capability to easily provide the quantitative measurement of the presence of some specific words that are more biased to have a positive connotation or a negative one. This would be enhanced when we visualize the result, and we can quickly come to some conclusion without performing any sentiment analysis. It could also give us an overview of authors from different newspapers who have the same tendency to use similar words when discussing a specific topic. Obviously, the result obtained was also removed from english stop words as they are commonly used but don’t convey much information.
### Bag of words
Using BoW will capture more meaningful information from the scraped data, as not only the frequency of the words will be taken into consideration but also their presence and absence in the documents, allowing us to use this information to be processed using clustering techniques to see how different newspaper view are related to each other when it comes to talking about specific topics.
### TF-IDF
As BoW does not take into consideration the term frequency within a document and it may be misled by the excessive use of stop words in the documents, having a third result obtained using TF-IDF may produce different results and enhance the quality of our analyses.
All of these three techniques were performed both in Python using the sklearn library and Scala, using the ml.feature library. The main purpose of this action was to see the difference in performance using a parallel approach over the sequential one but as it would be discussed more in the result sections, we did not find a great difference in time as our dataset was influenced by the web scraping limitations.
## Files
- `sentiment_prediction.ipynb`: This Jupyter notebook contains  code for sentiment analysis. 
It uses TensorFlow Keras to run the sentiment prediction on a pre-trained mode. The model is trained on Sentiment140 dataset with 1.6 million tweets. The pretrained model's URL is as followes: https://www.kaggle.com/code/abhineethmishra/twitter-sentiment-analysis/notebook
- `statistical_plot.R`: Reads the CSV from the sentiment analysis model and generate meaningful figures for data visualization and analysis.
- `bias_score.csv`: Bias score with column `{file,bias score}``.
- `time.csv`: running time of individual news files, the average time and total running time. With column `{file,time}`
- `color_hex.png`: coler hex for reference when choosing the color for visualization.
- `BoW.scala`: bag of word in scala
- `PythonPrepocessing.ipynb`: Text preprocessing(tf-idf), word count, bag-of-word
- `r_visualization.R`: another visualization in R
- `tf_idf_limit.scala`, `tf_idf.scala`: tf_idf in scala
- `word_count.scala`: word count in scala
- `time_performance.csv`: time comparason between spart and python
## Directory
- `pretrained_mode`: The pre-trained model and the tokenizer from the above repository.
- `cs4480_scraped_data`: The raw data scraped from various news media. The file name format is as NEWSTOPIC_MEDIA.txt.
- `results`: the results output by the sentiment_prediction.ipynb. The file name format is as NEWSTOPIC_MEDIA.csv.
- `plots`: the figures generated by statistical_plot.R.
- `neutralized_data` and `neutralized_result`: The news neutralized by ChatGPT and the corresponding result in the same CSV as above.
- `clustering_pca_results`: The result of clustering pca
- `data_preprocessing_result`: The results of BoW, tfidf and word_count, in CSV
- `data_scraping`: The data scraping's ipynb to scrap news using selenium
- `GPT_API`: The ipynb files to interact with ChatGPT by api and neutralized the news
- `word_count_top50`: The result of word count
- `wordcloud`: The resuls image of wordcloud
## Usage
1. Run `sentiment_prediction.ipynb` notebook to predict the sentiment. 
2. Run `bias_score.R` script generate the visualization figures on sentiment result on `sentiment_prediction.ipynb` 
3. Run `Bow.scala` to generate the Bag of word in scala
4. Run `PythonPreprocessing.ipynb` to generate the tf-idf, bag of word, and word counting, output the corresponding time and result.
5. Run `r_visualization.R` to generate some figures on R
6. Run `tf_idf_limit.scala` and `tf_idf.scala` to generate the tf_idf data in scala
7. Run `word_count.scala` to generate the word count on scala 
