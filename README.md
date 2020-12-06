
# Machine Learning for Algorithmic Trading Strategy


## Group 2:


**Changhong Ma**

Panther ID: 6169761


**Hongjing Wang** 

Panther ID: 6162083

# Introduction

Breakthroughs in artificial intelligence make headlines every day. The widespread adoption and powerful application of machine learning in the financial sector are far from the buzz of customer-facing businesses. There are few fields with as many raw and structured data as the financial industry, which makes it one of the predetermined use cases for the early success of the &quot;learning machine&quot; and continued great success. Our project applies machine learning to real-world environments in the real world beyond the examples.

Stock market forecasting is an attempt to determine the future value of company stocks or other financial instruments traded on financial exchanges. Successfully predicting the future price of the stock will maximize investors&#39; returns. This report proposes a machine learning model to predict stock prices. The main purpose of our project is to formulate and analyze specific machine learning methods suitable for strategy parameter optimization. The purpose of our prediction is to significantly shorten the calculation time without significantly reducing the quality of the strategy. We compare two different methods of Linear Regression (LR) and Recurrent Neural Network (RNN) to implement the optimal results. In our work, artificial neural networks technology has been used to predict the next day&#39;s closing prices. The financial data: the opening price, highest price, lowest price, and closing price of the stock are used to create new variables, which are used as input to the model.

We conduct experiments and conclude that the accuracy of the LR method is 63.63% and the accuracy of RNN is 0.083%. The results of RNN seem low, however, because even if we are very close to the answer that&#39;s not going to be considered accurate. We need to use the plot to declare the accuracy which will be shown in the below content. So our model can effectively predict the stock closing price.

All codes and datasets are available in our GitHub repository:
https://github.com/shiningMCH/ML\_Algo\_Trading

# Related Work

Algorithmic Trading is a process for executing orders utilizing automated and pre-programmed trading instructions to account for variables such as price, timing, and volume. Trading algorithms are executed to rely on computer programs to automate some or all elements of a trading strategy in the investment industry. Due to the volatility and non-linear nature of the financial stock market, accurately predicting stock market returns is a very challenging task. With the introduction of artificial intelligence and increased computing power, programmed forecasting methods have proven to be more effective in predicting stock prices.

Our team applies advanced machine learning into these strategy aims to make more efficient use of a rapidly diversifying the range of data to produce both better and more actionable forecasts, thus improving the quality of investment decisions and results.

Machine learning is a method that allows computers to search for patterns within data through tests, judging the results, then changing the tests according to the results. When applied to a trading strategy to choose the optimal portfolio, it&#39;s extremely powerful.

Get the data in place. A good source of financial time series is the API of the exchange. The scale of the data should be at least as good as the scale of the expected model, and finally, make predictions. Our data comes from Tiingo.com, a free API that allows us to query historical stock data of many companies including open prices, close prices, high prices, low prices, adjust prices, and volume for each data specified.

Split the data into supplementary sets for training, verification (for parameter adjustment, feature selection, etc.), and testing. Ideally, the test set should try to be as similar to the current market conditions as possible, and the verification and test sets should follow the same distribution.

# Dataset and Features

For this project, we decided to go with Tiingo.com to retrieve data. This is essentially a free and popular stock data API that allows us to query historical stock data. Tiingo has a massive database of many different stocks and data dating back several years and it supports many different forums and posts. So it can structure a query and read incoming data.

**Fetch Data**

First, we need to construct a new URL to get back to data with the parameters filled in. Generally, there are three parts to those API queries. Tiingo needs the stock name, the start date, and the end date to construct a URL as well as an authentication token or the API key. We need the base Tiingo URL and search for daily historical data of a particular stock. Then we punch the URL into a python program with the requests package and we can retrieve data. This retrieves the open and closed prices, the high and low prices, the adjusted closed price, and volume for each of the dates. Each of the data points is one day&#39;s worth of data and contains each of the above information. For example, we fetch APPLE&#39;s historical price data.
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/1.jpg"/></center>

<center>Figure 1: APPLE's historical price data</center>

We have collected data in the JSON file with every object contains all information about the stock price. And we just need the open and closed prices, the high and low prices, the adjusted closed price, and volume for each of the dates.

**Parse Data**

We need to parse the data out into separate datasets. For the most part, we are just interested in six pieces of data: date, close, open, heigh, low, volume. Our JSON data is in an array. Within the array, we have a bunch of these objects, so we can just use a for loop to iterate through this JSON data and fetch each of the object, and then get the pieces we want.

Then we manipulate the data that we have parsed previously as well as assigning labels and just overall building our training and testing sets. So we start by creating a function that will allow us to calculate the price differences. This will essentially be our labels as we are looking to see the differences between the price at the end of the day or closed price, and the open price the next day. We also create a function to help us assign these labels as well as just create our datasets. So we basically do all of the fetching, parsing, and data assignment in one function and that allows us easily create our training and testing datasets.
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/2.jpg"/></center> 
<center> 
Figure 2: APPLE&#39;s stock price graph
</center>
We can see that the blue line is the open prices, and the orange line is the close prices.

**One-Hot Encoding**

The label is the difference between the next day&#39;s open and the current day&#39;s close price. And we will use all of today&#39;s data to predict the next day&#39;s opening price and whether it goes up or down. So we are not looking for an exact dollar figure of what the price is going to be. We are looking for is whether the price will go up or will it go down. So we can assign 1 if the price goes up or 0 if the price goes down. Here our team uses One-hot encoding method. This means that instead of doing a 1 or a 0 if the price goes up or down respectively. We actually use a [1, 0] if the price goes up, and [0, 1] if the price goes down. We try to classify based on today&#39;s data whether the price will belong to the increased category or the decrease category. We do not aim to achieve a specific price itself.

We need to loop through the open prices and the closed prices and comparing them. We skip out on the very last closed point and skip out on the very first open point because we try to compare the current close to the next data points open. Then we can see if the open price on the next day is going to be greater than or equal to for that matter of the closed price of the current day. So if it&#39;s greater than or equal to we are going to say it&#39;s increased, and so we will append a [1, 0]. If it decreases then we append [0, 1]. This can help us fit into the model.

The one-hot encoding method works well when we deal with classification problems. Realistically for this problem we could just use integer encoding, for example, increasing is 1, and decreasing is 0. But it has a shortcoming that it makes fitting stuff into the model a litter more complicated. Here is the Visualization graph of the integer encoding method.
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/3.jpg"/></center>
<center>
Figure 3: Integer encoding method
</center>
# Methods

**Linear Regression**

We begin the basis of the linear regression model. Essentially this is based on Y=wx+b. We try to fit a line of best fit through the data. In our case, we are taking one-hot encoding as inputs. [1, 0] standards for the stock goes up the next day and [0, 1] means stock goes down the next day.

First, we build a lost function. We want to average everything out and then get the sum of everything to get the total loss. We add together all of the components of the lost function to try to find the average or the minimum average of the answers. The logits equals are what the model is outputting and labels is the correct answers, which exactly means the y\_train and y\_test. We try to find the difference between these two that gives us the loss.

Next we need an optimizer. We use AdamOpimizer or GradientDescentOptimizer in tensorflow. We have used a couple of different optimizer to test and draw the conclusion . We do a learning rate of 0.01. We also minimize loss at the same time rather than creating an extra step. So this optimizer is going to try to minimize our loss which is going to be essentially the difference between our models outputs and the correct outputs/labels.

**Recurrent Neural Network**

Here we use Keras library. Keras is a library used to build neural networks for machine learning models. It is a higher-level API build on top of some machine learning backend. So Keras needs a backend framework for machine learning computation. There are actually many we can choose from and the default is TensorFlow. Here we use TensorFlow as the backend. Then we add layers to a model one at a time using layer objects.

There are two types of models, sequential and functional models. The sequential models connect a layer to the next based on the order in which we added them. Functional models are more flexible and allow us more control over how to connect the layers.

Why we use Keras? There are four reasons. First, model and layer objects make it much easier to build machine learning models. Tensorflow actually simplifies the overall process of building a machine learning model, but it&#39;s still not easy to use especially if we don&#39;t have too much idea. Keras take away this trouble, which basically provides us all of these model and layer objects that we can just call the instances and it does all the tasks for us. Second, we no longer need to build up layers by creating and connecting variable and operation nodes. We have used the linear regression model in the first experiment. In that model, we create the nodes individually. If we do that in Keras, we just need to build a single layer using a single object. Third, we can also specify loss and optimize the functions with compile function in Keras. So we don&#39;t have to create an extra train step and choose the optimizer to tell it to minimize the loss. We can basically just use the Keras compile function on the model object and then we get to pass in our activation function learning rate, epochs, etc. Finally, we can use the &#39;fit&#39; function to train the model and the &#39;predict&#39; function to use the trained model.

RNN what we call recurrent neural networks is looped networks that allow information or state to persist between runs. It doesn&#39;t operate on a fixed number of layers but rather cycle the input through a single layer many times, combining the current state with new input with each cycle. There is a bunch of values free to the nodes in each of the layers and we run it through each of those layers. That gives us a fixed number of steps and it will produce some sort of output.

Then we modify those values to get closer to the target. With RNN we just have 1 or 2 layers, and we run the layers over and over again. Each of the LSTM cells will have a state that is remembered from the previous cycle through and it will combine that state with the new input to produce some other outputs. LSTM cells are for series and sequences of data as they can &#39;remember&#39; previous values, especially for time series. And Extremely small values sometimes don&#39;t change in traditional neural, which can get almost dead neurons that actually cause training to stop. LSTM cells can fix this problem.

# Experiments

**Linear Regression**

We train and test our model on the datasets we create. So we start by building a function to measure accuracy. It can compare the predicted value or the correct value to the actual moral output. If they match up we add 1 to the accuracy score. If they don&#39;t match up then we just add nothing. We take an average and find the overall accuracy. We also run the model through a few months&#39; worth of data to train it. And here we just use one month to test. After trying, we find out that the best results appeared that when we use a month&#39;s worth of data to test.

We find that we get good results if we run between 10 and 20000 iterations. Anything less than that will cause suboptimal results and anything more than that just really doesn&#39;t make a difference.

After we train the model then we measure the accuracy. We don&#39;t really care too much about the accuracy of the training set because likely the training set is lower in the accuracy anyway. So we run the accuracy measure on the testing set.

We create a function called measure\_accuracy. It can take in an actual answer and an expected answer. The actual answer is the model&#39;s output and the expected answer is the correct answer. We iterate through the actual expected and find out the correct number that means we predict successfully. Then we calculate the average correct number. The result will be a decimal number and we just need to multiple 100 to get the percentage.

The result of our accuracy is 63.63 percent. What we can do to improve this? First, we can change the optimizer to seek better results. Some other improvements we can make would be to run it through more epochs. We found that doing more epochs doesn&#39;t really make much of a difference. But in some models that will help some more. The other thing to do is lower the learning rate. But usually, we have to up the number of epochs as well. We still need to create a more complex model.

**Recurrent Neural Network**

We fetch the data of JPMorgan Chase. below is the close price from 01/2014 to 11/2020.
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/4.jpg"/>  
</center>
<center>
Figure 4: JPMorgan Chase close stock price
</center>
Then we use the MinMaxScaler function to normalize the data so that all the values are between 0 and 1. We try to eliminate the difference between the lower price and the higher price. Because we are not interested in using this data for the prices individually. We try to assess a trend based on how the prices have changed over time.

We use 70% of the data as a training set and 30% as a test set. Then we build a model with a model object. And then we just add in our layers. We create the layers first and then add the input one by one. After that, we need to add a dense layer. The dense layer is simply going to take 1 as the outputs. That&#39;s because we just need the price to be outputted. And set the activation as &#39;sigmoid&#39;. We finish the model built.

Then we need to specify our loss and optimizer functions which we can implement by the compile function. We use &#39;mean\_squared\_error&#39; as loss and &#39;adam&#39; as the optimizer.
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/5.jpg"/>
</center>
<center>
Figure 5: Model summary
</center>
As shown above, the model summary, help to tell how the model is structured overall. It tells us which layers we are using, the output shape at each layer, etc.

Then we train it with the fit function by passing the x\_train and y\_train. We still need a score to show how well the model performed. The results are flowing:
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/5-1.jpg"/>
</center>
The loss we get is pretty low. However, the accuracy is low. That because even if we are very close to the answer that&#39;s not going to be considered accurate. For example, if our model can predict to 10 and the prices to 11. It is very close to the correct answer. So this model is actually working quite well.
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/6.jpg"/>
</center>
<center>
Figure 6: JPMorgan Chase&#39;s prediction graph
</center>
For figure 6, the blue line is the real stock price we read in the beginning, the orange is the 70% training set and the green line is the 30% testing set. The training may a little off the blue original data but for the most part, actually follow the trends well.

# Summary, Future Work

Our first experiment uses basic linear regression to do the initial examination. Our main focus is actually on the second experiment. Keras is really helpful when we construct deep neural network models. In our project, we use a recurrent neural network and extend it to LSTM cells. LSTM cells help us to maintain a status and they give us more accurate results over time-series data. We split our data into a 70/30 status. We also compile it and learn how to use the fit function. We use the fit function to train the model and the predict function to test the model. And finally, we interpreted the results and plot the result.

Start here, our next step work will focus on improving the model. Our model makes pretty good predictions now, but there is always room for improvement. We could learn to take other factors into accounts such as global news or the sentiment of people. For example, if we use the Tesla.Inc&#39;s stock price to train the model, the result is unsatisfactory (Figure 7). Tesla&#39;s stock price has risen wildly in recent years, rising by more than 1,000% in more than a year. The factors are very complicated, and our model does not consider these factors. We can see the green line is far away from the real data.
<center>
<img src="https://raw.githubusercontent.com/shiningMCH/ML_Algo_Trading/master/readme-assets/7.jpg"/>
</center>
<center>
Figure 7: Tesla&#39;s RNN method prediction results.
</center>
As for now, our price prediction model is just based on what the close prices are. If we can add the global news or sentiment, then we have a way to increase our model and to account for all possible changes rather than just dwelling on the numbers.

# References

**Academic Research:**

The stock market is now run by computers, algorithms and passive managers, Economist, Oct 5, 2019

[Algorithmic trading review](http://doi.org/10.1145/2500117), Communications of the ACM, 2013

Kumar, Manish, and M. Thenmozhi. (2006) &quot;Forecasting stock index movement: A comparison of support vector machines and random forest&quot; In Indian institute of capital markets 9th capital markets conference paper

[Financial Time Series Forecasting with Deep Learning : A Systematic Literature Review: 2005-2019](http://arxiv.org/abs/1911.13288), arXiv:1911.13288 [cs, q-fin, stat], 2019

**Books:**

Machine Learning for Algorithmic Trading - 2nd Edition

[Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086), Marcos Lopez de Prado, 2018
