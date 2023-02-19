
## Movie Recommendation System 

# Problem formulation
In this project, we will be building our recommender system, similar to the one used by Netflix. Some of the business questions which we will seek to answer include the following:
1.	Given a user’s history or movie preferences, which movie is the user likely to be interested in?
2.	Should we adopt a binary class approach (recommend or not recommend) or a multi-class approach (ratings - 1, 2, 3, 4, and 5)
3.	If there is more than one movie to recommend, what ranking system should be used to determine the order of the movie list?
Data sets:
Using TMDB Data Set
![image](https://user-images.githubusercontent.com/47840840/219969754-b20359e4-2963-47da-9e8a-fab479693c3e.png)
Saved Models:  "SVDppModel.pickle" and "modelSVDppLoo_1M.pickle".

The [MovieLens] data set

![image](https://user-images.githubusercontent.com/47840840/219969777-9ff692cb-7de3-4467-9b67-492dacf2f5a9.png)

![image](https://user-images.githubusercontent.com/47840840/219969804-d9e74f7d-5e3c-49c8-8260-febb76b15475.png)

# Data Preparation

1.	Merging TMDB Data Sets

![image](https://user-images.githubusercontent.com/47840840/219969825-4bab1d37-efa9-4089-8407-52ad43b1e67a.png)

2.	Cleaning TMDB Data Set 

![image](https://user-images.githubusercontent.com/47840840/219969833-2e1ecd74-d2b1-492f-bf96-19dfbb536fef.png)

![image](https://user-images.githubusercontent.com/47840840/219969834-5389c97d-557c-4284-9462-d3e772e59c37.png)

3.Check null values in rating data and scaling it 
![image](https://user-images.githubusercontent.com/47840840/219969843-9e2d5751-ceed-4966-a012-27987270623b.png)

![image](https://user-images.githubusercontent.com/47840840/219969849-69347d72-aea2-4a1d-8657-be44e375f767.png)

![image](https://user-images.githubusercontent.com/47840840/219969850-1efa1b7f-f5f1-48a0-8099-a8eb5723f7c3.png)

# Methods:
Demographic Filtering :
calculate the mean which is our c variable
![image](https://user-images.githubusercontent.com/47840840/219969875-d317a8ea-125b-4888-8043-0339171170cf.png)
Calculate the min votes which is m
![image](https://user-images.githubusercontent.com/47840840/219969886-b2e99862-f488-498e-b01d-b4436c2e21bd.png)
![image](https://user-images.githubusercontent.com/47840840/219969889-e7abdd49-f59d-4bb2-9b3b-c7a06f300886.png)
![image](https://user-images.githubusercontent.com/47840840/219969908-5e65a551-465d-48ce-81ec-50a4c3cbfaa4.png)
![image](https://user-images.githubusercontent.com/47840840/219969912-45e3adc5-cbd7-4d2b-a7c6-175e2be415b3.png)
Define score for the movies
![image](https://user-images.githubusercontent.com/47840840/219969921-fac5c79a-2ae8-44be-9efc-0024aae563ee.png)
Showing the top 10 high scours movies
![image](https://user-images.githubusercontent.com/47840840/219969935-f242d410-5ded-4013-bc59-ab6b7656d838.png)
Showing the top 10 popular movies
![image](https://user-images.githubusercontent.com/47840840/219969941-dd117856-3c71-4cad-a687-506fbb1d5103.png)

# Content Based Filtering
![image](https://user-images.githubusercontent.com/47840840/219969976-ebf1170d-7585-4606-89c9-3a206bd451b6.png)
Finding similarity from the movie title and overview only 
![image](https://user-images.githubusercontent.com/47840840/219969985-24953951-6f48-4133-b3f1-b22235e43218.png)
Finding similarity from the Credits, Genres and Keywords
![image](https://user-images.githubusercontent.com/47840840/219969992-31203275-cd30-498b-a2e6-16f8b1688a58.png)
# Text Feature Engineering  
TF-IDF Vectorizer
![image](https://user-images.githubusercontent.com/47840840/219970005-731062ab-367f-4ea2-9078-b215b13477dc.png)
CountVectorizer instead of TF-iDF:
This is because we do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies. It doesn't make much intuitive sense.
![image](https://user-images.githubusercontent.com/47840840/219970024-77922990-eed9-4a1e-92fb-01d95186516f.png)

# Collaborative Filtering
We used some algorithms in surprise library:

-	Matrix Factorization-base models: SVD, SVDpp 
-	Classification: KNNBaseline , KNNBasic, KNNWithMeans, KNNWithZScore
-	Cluster: CoClustering
![image](https://user-images.githubusercontent.com/47840840/219970048-ec3d9837-c1ec-4870-af28-9f0b11fda22f.png)
![image](https://user-images.githubusercontent.com/47840840/219970056-b7c5e469-b30f-4b65-80f9-faed095b86c1.png)
![image](https://user-images.githubusercontent.com/47840840/219970063-c1f3e9fa-c670-4f33-b27a-e5705187c552.png)
Split dataset After Scaling to train champoin model:
![image](https://user-images.githubusercontent.com/47840840/219970076-2271405e-ead8-41ce-844e-30539eb9b2e6.png)
We chose the champion model (SVDpp):
The reason is the test_rmse of SVDpp model is the smallest value, but it is taking a lot of time so we will apply GridSearch CV to get the best values of parameters in SVDpp Model:
![image](https://user-images.githubusercontent.com/47840840/219970083-406cffab-49c4-4c23-be9f-474809e68a78.png)
After training the champion model (SVDpp), we saved it in ‘SVDppModel.pickle’.
So we shouldn’t train SVDpp model again.
![image](https://user-images.githubusercontent.com/47840840/219970088-176df808-a4c5-40f1-bf67-81ffed54612a.png)
# Evalution on and all All models and champion model (SVD):
![image](https://user-images.githubusercontent.com/47840840/219970111-54711f73-4c95-4b49-a6f1-2c7cc076cdd3.png)
Load champion Model (SVDapp)
![image](https://user-images.githubusercontent.com/47840840/219970130-b72f4b7a-2385-4d19-b97c-b2e73ed4a843.png)
![image](https://user-images.githubusercontent.com/47840840/219970134-4440553a-2d6d-4fec-a697-029fe4b1a3cc.png)
By predicting by SVDpp, we can determine the best and worst predictions. 
![image](https://user-images.githubusercontent.com/47840840/219970144-f18badab-221b-4802-abd6-4c53dd66a171.png)
![image](https://user-images.githubusercontent.com/47840840/219970148-12ec5c44-371b-4a7b-9fcb-31b51f97f0ec.png)
In the lift table, the best predictions are not lucky guesses.  Because Ui is anywhere between 13 to 104, they are not really small, meaning that significant number of users have rated the target movie.
In the lift table, the best predictions are not lucky guesses.  Because Ui is anywhere between 13 to 104, they are not really small, meaning that significant number of users have rated the target movie.
Example to test champion model SVDpp: 
Those are the liked movies for the user with id 26  
![image](https://user-images.githubusercontent.com/47840840/219970160-fe962900-9a4e-466f-9b7f-fcdab7c45a86.png)
# Error Analysis:
![image](https://user-images.githubusercontent.com/47840840/219970169-a39da093-ebd4-4cc4-bc67-03a017898343.png)
![image](https://user-images.githubusercontent.com/47840840/219970170-7f44d916-f705-4676-a046-20c518dc0086.png)
We removed one of these movies (Leave-One-Out cross-validation).
To evaluate top-10, we used hit rate, that is, if a user rated one of the top-10 we recommended, we consider it is a “hit”.
![image](https://user-images.githubusercontent.com/47840840/219970177-d578305c-b8fb-4671-88e8-bf856da3088b.png)
We can save the training model of Leave-One-Out cross-validation in ‘modelSVDppLoo_1M.pickle’, so we shouldn’t train it.
![image](https://user-images.githubusercontent.com/47840840/219970182-1e366e0c-4f1e-4aab-ad40-c4f9077f4984.png)
![image](https://user-images.githubusercontent.com/47840840/219970186-313bb9c5-625c-4b29-b58b-8bf8892034a3.png)
We used Hit Rate by Rating Value on ‘modelSVDppLoo_1M.pickle’
By using the predicted rating values, we can deconstruct hit rate. In an ideal world, we would be able to predict how well-liked a movie will be by its audience.
![image](https://user-images.githubusercontent.com/47840840/219970194-09baf487-3248-459d-88b2-14183cdda521.png)
Our hit rate distribution matches my expectations completely.  The rating score 5 has a substantially better hit rate than the others. Consequently, our champion model (SVDpp) can anticipate it correctly.

#Visualization of results, and graphical intuition and analysis
Connecting our code with Dialog flow chatbot via NGROK
First, let’s test content-based recommendation which is recommending a movie similar to a movie you ask for, and work to analyze the movie metadata: cast, keywords, director, and genres. and present a movie similar to this data.
![image](https://user-images.githubusercontent.com/47840840/219970223-c9cfe6a3-dffc-468e-9be1-efd3fb8430a9.png)
![image](https://user-images.githubusercontent.com/47840840/219970226-35327cab-011e-4182-8128-e5e2edee34f1.png)
![image](https://user-images.githubusercontent.com/47840840/219970227-ffc91736-9d29-4721-af6e-1bc1a5a3cecd.png)
As we see in this figure there is no response to any question. It extracts any movie name that we have provided in the entity names and then communicates directly to the host to get a response from It. 
![image](https://user-images.githubusercontent.com/47840840/219970242-9f575efe-2acc-4edf-b01c-b49ab042bbff.png)
![image](https://user-images.githubusercontent.com/47840840/219970247-685c494c-2768-4a53-91ef-6a94c3ba104c.png)
This entity contains some examples of our data set to catch any movie in our dataset immediately
Second, let’s test collaborative filtering by knowing users what would like to watch according to their history 
![image](https://user-images.githubusercontent.com/47840840/219970253-b948732e-c719-4bf3-aaf6-6ae5b45ed7ef.png) ![image](https://user-images.githubusercontent.com/47840840/219970263-1f96dd2a-9515-4683-b781-c847bd3e5079.png) ![image](https://user-images.githubusercontent.com/47840840/219970269-4bcf1571-40c2-45d3-9056-9aa59e7c550e.png)
In this feature, the chatbot extracts the number of the user and gets preferences for him

# Innovativeness 
We applied every recommendation like non-personalized and personalized, starting with the non-personalized, which is a probability estimation similarity using the IMDB formula. Then we jumped into the personalization and we applied to content based on the movie’s output. As a feature, we will add the score from the weighted rate formula that we calculated to improve the recommendation. Then we apply the content with our Metadata ("cast, crew, directory, and keywords") by using cosine similarity with TFiDF to make the recommendation more accurate. Also, we used a technique called Collaborative Based Filtering, which is an approach where user ratings come into account, and hence there can be different outputs possible on the basis of the reviews given to the items or movies in focus. We compared even more algorithms (SVD, SVDpp, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, and CoClustering) to select the best champion model (SVDpp). We applied GridSearch to reduce the training time for the SVDpp problem. This champion can extract movies that are more suitable for a particular user. We integrated these methods that applied more models such as cosine similarity and SVDpp with a chatbot that can recommend top movies to users by some previous standards.

