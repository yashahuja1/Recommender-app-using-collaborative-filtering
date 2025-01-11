Movie Recommender System using Singular Value
Decomposition

Abstract—Movie recommendation systems play a crucial role
in enhancing user experience by predicting personalized pref-
erences. This study explores the implementation of a movie
recommender using Singular Value Decomposition (SVD) and
compares its performance with Alternating Least Squares (ALS)
and Neural Collaborative Filtering (NCF). Leveraging user
ratings data, SVD employs matrix factorization techniques to
model user-item interactions, offering an effective balance be-
tween simplicity and performance. SVD achieved an RMSE
of 0.9015, outperforming both ALS (RMSE: 0.9229) and NCF
(RMSE: 0.9502). Its ability to capture latent factors in user-movie
interactions while maintaining computational efficiency makes
SVD particularly promising for handling sparse and diverse data.
SVD’s superior performance positions it as a robust approach
for next-generation recommendation systems.

Index Terms—Neural Networks, Recommender System, Machine Learning, Data Analysis, Collaborative Filtering

Similarly, NCF employs deep learning to model intricate
relationships between users and items; however, it often requires substantial computational resources and extensive hyperparameter tuning. Additionally, NCF can struggle with data
sparsity, especially when user feedback is limited, resulting in
less reliable predictions for less popular movies.

In contrast, Singular Value Decomposition (SVD) excels in
uncovering latent factors in user-movie interactions through
matrix factorization. Its ability to efficiently capture underlying
patterns allows SVD to provide robust recommendations even
in sparse datasets. This study aims to leverage SVD’s strengths
while addressing the limitations of ALS and NCF, ultimately
enhancing the accuracy and relevance of movie recommendations.

I. INTRODUCTION

In the era of information overload, recommendation systems
have become indispensable tools for filtering and personalizing vast amounts of data to meet user preferences. In
the entertainment industry, where users are inundated with
options, movie recommendation systems play a crucial role
in delivering tailored suggestions, enhancing user satisfaction,
and driving engagement. These systems analyze user behavior
and movie attributes to predict which films users are most
likely to enjoy, thereby bridging the gap between content
abundance and individual preferences.

A. Problem Definition:

In the landscape of movie recommendation systems, traditional techniques such as Alternating Least Squares (ALS) and
Neural Collaborative Filtering (NCF) present certain limitations that can impact their effectiveness. ALS, while efficient
in handling large datasets, relies on linear assumptions that
may not adequately capture complex user-item interactions,
particularly in scenarios with non-linear relationships. This
limitation can lead to suboptimal recommendations for users
with diverse preferences.

B. Objective:

The objective of this study is to design and evaluate a
movie recommendation system using Singular Value Decomposition (SVD), a powerful matrix factorization technique,
and compare it with Alternating Least Squares (ALS) and
Neural Collaborative Filtering (NCF). SVD leverages latent
factor modeling to capture underlying patterns in user-item
interactions, offering a balance between computational efficiency and predictive accuracy. This study aims to compare the
performance of SVD against ALS and NCF, using Root Mean
Squared Error (RMSE) and Mean Absolute Error (MAE)
as the evaluation metrics. The study seeks to analyze the
distribution of user ratings, temporal trends, and other dataset
characteristics to provide insights into the behavior of the
models and their recommendations.

C. Significance:

The significance of this study lies in its comprehensive
evaluation of different recommendation system approaches,
particularly highlighting the effectiveness of Singular Value
Decomposition (SVD). While Neural Collaborative Filtering
(NCF) employs deep neural networks to model complex interactions, this study demonstrates that SVD, a matrix factorization technique, outperforms both NCF and Alternating Least
Squares (ALS) in handling diverse and sparse datasets. SVD’s
ability to capture latent factors allows it to identify subtle
patterns in user preferences and movie attributes, addressing
real-world challenges such as the cold-start problem and
imbalanced data distribution.

The insights gained from this study are both academically
valuable and practically applicable to industries reliant on
personalized content delivery. By highlighting the strengths
of SVD compared to ALS and NCF, this work provides a
roadmap for implementing robust and accurate recommendation systems. Such advancements are particularly relevant
in the context of rapidly growing streaming platforms, where
effective recommendations can significantly enhance user retention and satisfaction.

In summary, this study underscores the importance of
carefully evaluating different approaches in recommendation
systems. It demonstrates how SVD can effectively address
the limitations of both traditional methods and more complex
neural network-based approaches, while opening avenues for
future innovation in movie recommendations.

II. LITERATURE REVIEW

A. Overview of Traditional Recommendation Techniques

Recommendation systems are commonly classified into
three categories: content-based filtering, collaborative filtering, and hybrid methods. Content-based filtering relies on
item attributes (e.g., genre, director) to recommend movies
similar to those the user has previously liked. However, this
approach is limited by its inability to recommend diverse items
outside a user’s historical preferences. Collaborative filtering
(CF), on the other hand, leverages user-item interaction data,
such as ratings, to find patterns of similarity either between
users (user-based CF) or items (item-based CF). Among CF
methods, matrix factorization techniques like Singular Value
Decomposition (SVD) and Alternating Least Squares (ALS)
have gained prominence. These methods decompose the useritem interaction matrix into latent factors representing user and
item characteristics. While effective, they assume a linear relationship between these factors, limiting their ability to model
complex user behaviors. Additionally, they are prone to data
sparsity issues and struggle to incorporate side information,
such as temporal trends or metadata.

B. Matrix Factorization Techniques: Focus on SVD

Singular Value Decomposition (SVD) has emerged as a
powerful matrix factorization technique in collaborative filtering. SVD decomposes the user-item interaction matrix into
latent factors representing user and item characteristics. Unlike
simpler linear models, SVD can capture complex patterns in
user-item interactions through these latent factors. Its effectiveness in handling sparse data and computational efficiency
have made it a popular choice in recommendation systems.
While SVD assumes a linear relationship between factors, its
ability to reduce dimensionality and uncover hidden features
often leads to robust performance across various datasets.

C. Neural Collaborative Filtering and Deep Learning Approaches

Neural Collaborative Filtering (NCF) represents an attempt
to integrate deep learning into recommendation systems. NCF
replaces the linearity of matrix factorization with non-linear
neural networks, aiming to model intricate user-item relationships. At its core, NCF embeds users and items into dense vectors and uses a Multi-Layer Perceptron (MLP) to
learn their interactions. This approach theoretically allows
NCF to capture non-linear dependencies, making it particularly
suited for handling diverse and sparse data. However, the
practical performance of NCF compared to traditional methods
like SVD can vary depending on the specific dataset and
application.

D. Related Work

The exploration of deep learning in recommendation systems has gained momentum in recent years. He et al.’s
”Neural Collaborative Filtering” (2017) introduced NCF as a
novel framework that generalizes matrix factorization under
a deep learning perspective. The authors demonstrated that
NCF significantly outperformed traditional CF models like
SVD in capturing user-item interactions, particularly in sparse
datasets. Their model architecture includes a Generalized
Matrix Factorization (GMF) component combined with a non-linear MLP for enhanced representation learning.

Similarly, Xue et al.’s ”Deep Matrix Factorization Models
for Recommender Systems” (2017) investigated the integration
of deep neural networks into matrix factorization. They proposed models such as DeepMF, which incorporate additional
layers into traditional MF to improve the model’s ability
to learn non-linear patterns. Their findings underscored the
potential of deep learning to overcome the limitations of
standard MF, particularly in terms of flexibility and accuracy.
Recent comparative studies have shown mixed results when
comparing traditional matrix factorization techniques like SVD
with neural network-based approaches. For instance, Dacrema
et al. (2019) in ”Are We Really Making Much Progress?
A Worrying Analysis of Recent Neural Recommendation
Approaches” found that well-tuned traditional algorithms often
outperform more complex neural models in many scenarios.
This highlights the continued relevance and effectiveness of
methods like SVD in practical recommendation systems.

Both earlier and more recent studies highlight the ongoing
debate in the field of recommendation systems. While neural
network approaches like NCF offer theoretical advantages
in modeling complex patterns, traditional methods like SVD
continue to demonstrate robust performance, especially when
well-optimized for specific datasets. This underlines the importance of thorough empirical evaluation in choosing the most
suitable approach for a given recommendation task.

III. UNDERSTANDING THE DATASET

A. Source and Description

The dataset used for this study is sourced from The Movies
Dataset, hosted on Kaggle. This extensive dataset is widely

utilized for building and evaluating recommendation systems
due to its rich user-item interaction data and associated metadata. It comprises several files, the most notable being:

1) Ratings.csv: Contains over 26 million ratings, capturing
user preferences for movies. Each entry specifies a userId,
movieId, rating (on a scale of 0.5 to 5), and a timestamp.

2) Movies.csv: Provides metadata about movies, including

movieId, title, and genres.

3) Links.csv: Links the movieId in the dataset to external

sources such as IMDb and TMDb.

4) Tags.csv: Contains user-generated tags for movies, offering additional descriptive features.

5) Credits.csv and Keywords.csv: Include cast, crew, and
keyword information for movies, providing opportunities to
incorporate content-based features. For this study, the focus is
primarily on the Ratings.csv and Movies.csv files. The Ratings.csv serves as the backbone for collaborative filtering by
detailing user-movie interactions, while Movies.csv provides
contextual data for analysis and visualization.

![Distribution of movie ratings](https://github.com/yashahuja1/Recommender-app-using-collaborative-filtering/blob/main/Images/distribution_of_movie_ratings.png)\
Fig. 1. Distribution of Movie Ratings

B. Preprocessing

Given the large size of the Ratings.csv file, containing over
26 million rows, preprocessing was performed using Apache
Spark, a distributed computing framework optimized for handling large-scale datasets. Spark’s DataFrame API enabled
efficient loading, cleaning, and transformation of the data. Key
preprocessing steps included:

1) Data Cleaning:
• Missing or inconsistent entries Movies.csv were filtered out.

in Ratings.csv and

• Entries with invalid or incomplete user IDs, movie IDs,

or ratings were removed.
2) Filtering Rare Interactions:
• Users with fewer than 10 ratings and movies with fewer
than 20 ratings were excluded. This step reduced sparsity
and ensured meaningful interactions in the dataset.

3) Timestamp Conversion:
• The timestamp column in Ratings.csv was converted to
a human-readable format, enabling temporal analysis of
user activity trends.

4) Merging Data:
• The Ratings.csv and Movies.csv files were joined using
the movieId column to map movie metadata with user
ratings.

5) Standardization:
• The ratings were standardized to a consistent scale and
normalized, ensuring compatibility with the model’s input
requirements.

6) Negative Sampling:
• Implicit feedback was generated by sampling user-movie
pairs with no recorded interactions, labeled as negative
examples. This step balanced the dataset for Neural
Collaborative Filtering, which requires both positive and
negative samples for training.

7) Train-Test Split:
• The dataset was split into training (80%) and testing
(20%) subsets, ensuring no user-movie pairs in the test
set overlapped with those in the training set.

By leveraging Spark, preprocessing was both efficient and
scalable, allowing the handling of millions of records without memory constraints. This robust preprocessing pipeline
ensured that the data was clean, consistent, and ready for use
in the recommender system.

IV. METHODOLOGY

This section outlines the design and implementation of the
movie recommender system using Singular Value Decomposition (SVD). It delves into the model’s architecture, the training
process, and the implementation details.

A. Singular Value Decomposition (SVD) Model

SVD is a matrix factorization technique that decomposes
the user-item interaction matrix into latent factors representing
user and item characteristics. This method is effective for
collaborative filtering in recommendation systems.

B. Model Architecture

1) Input Data:
• The model takes user IDs, movie IDs, and ratings as

input.

• Data is loaded from a Pandas DataFrame containing

’userId’, ’movieId’, and ’rating’ columns.

• Ratings are on a scale from 0.5 to 5.
2) SVD Parameters:
• Number of factors (nf actors): 100
• Number of epochs (nepochs): 20
• Learning rate (lrall): 0.005
• Regularization term (regall): 0.02

C. Training Process

• The dataset is split into training (80
• The SVD model is trained on the training set using the

specified parameters.

• The model learns latent factors for users and items during

the training process.

D. Evaluation Metrics

1) Error Metrics:

• Root Mean Squared Error (RMSE)
• Mean Absolute Error (MAE)

2) Ranking Metrics:

• Precision@10
• Recall@10

E. Recommendation Generation

• Top-N recommendations are generated for a user by

predicting ratings for all unseen movies.

• Predictions are sorted to select the top N movies with the

highest predicted ratings.

F. Implementation Details

• The model is implemented using the Surprise library in

Python.

H. Training Process

1) Hyperparameters: The following hyperparameters were

tuned for optimal performance:

• Number of factors: The size of the latent vectors for users

and movies (default: 100).

• Learning rate: Controls the step size in gradient descent

(default: 0.005).

• Regularization term: L2 regularization applied to model

parameters (default: 0.02).

• Epochs: Number of complete passes through the dataset

(default: 20).

2) Data Preparation: SVD requires explicit feedback data
in the form of user ratings. The dataset is prepared as follows:
• Ratings are normalized to ensure consistency across user

reviews.

• The dataset is split into training (80%) and testing (20%)
sets using a stratified sampling approach to maintain
distribution.

• PySpark is used for initial data processing before converting to a Pandas DataFrame for model training.

• Custom functions are implemented for calculating Precision and Recall at k, and for generating top-N recommendations.

3) Optimization: The model

is trained using Stochastic
Gradient Descent (SGD) with a specified learning rate. The
optimization process involves minimizing the Mean Squared
Error (MSE) loss function.

4) Evaluation Metrics: During training, the model is eval-

G. Loss Function and Optimization

1) Mean Squared Error (MSE) Loss: The SVD model in
this study uses explicit feedback in the form of user ratings.
The model optimizes the Mean Squared Error (MSE) loss,
which measures the average squared difference between the
predicted ratings and the actual ratings.

2) Regularization: To prevent overfitting, the SVD implementation includes regularization. The ‘reg all‘ parameter (set
to 0.02 in this case) applies L2 regularization to all model
parameters. This helps control the model’s complexity and
improve its generalization to unseen data.

3) Stochastic Gradient Descent (SGD): The model is optimized using Stochastic Gradient Descent. The learning rate
(‘lr all‘) is set to 0.005, which controls the step size at each
iteration while moving toward a minimum of the loss function.

Fig. 2. User Ratings

uated using metrics such as:

• RMSE: For numerical validation of predictions.
• MAE: To measure the average magnitude of errors in predictions.
• Precision@10 and Recall@10: For assessing ranking accuracy.

I. Implementation

1) Tools and Frameworks: The model was implemented in

Python using the following libraries:

• Surprise: For building and training the SVD model.
• Pandas: For data manipulation and preprocessing.
• Matplotlib/Seaborn: For visualizing trends in user activity and ratings.

2) Key Code Components:
• Data Preprocessing

– Performed using Pandas to handle the ratings dataset

efficiently.

– Filtering, normalization, and splitting into training

and testing datasets were completed at scale.

• Model Definition:

– Built using the Surprise library, defining parameters
such as number of factors, learning rate, and regularization terms for the SVD model.

• Training Script

– The SVD model was trained on the training set with

specified hyperparameters.

– Early stopping was not applicable as SVD typically

trains over a fixed number of epochs.

• Evaluation

– The model’s predictions were compared against the
test set, calculating RMSE, MAE, and ranking metrics such as Precision@10 and Recall@10.
By combining efficient data preprocessing with the robust capabilities of the Surprise library for SVD, this implementation
effectively scales to handle large datasets while delivering
accurate recommendations.

V. MODEL PERFORMANCE

A. Evaluation Metrics

To evaluate the performance of the movie recommender
system based on Singular Value Decomposition (SVD), a
combination of prediction and ranking metrics was employed.
Prediction metrics such as Root Mean Squared Error (RMSE)
and Mean Absolute Error (MAE) quantify the accuracy of
predicted ratings, providing insights into how closely the
model’s predictions align with actual user ratings.

Ranking metrics,

including Hit Ratio (HR), Normalized
Discounted Cumulative Gain (NDCG), and Mean Reciprocal
Rank (MRR), offer additional insights into the quality of the
ranked recommendations.

• **Hit Ratio (HR)** measures whether a relevant item
appears in the top-K recommended items for a user. If
a relevant movie is present in the list, HR for that user is
1; otherwise, it is 0.

• **Normalized Discounted Cumulative Gain (NDCG)**
takes this one step further by evaluating the ranking
order within the top-K recommendations, where items at
higher ranks contribute more to the score. This metric is
particularly useful when users care about having the most
relevant recommendations appear first.

• **Mean Reciprocal Rank (MRR)** computes the reciprocal of the rank of the first relevant item in the recommendation list. It reflects how early in the recommendation
list the correct result appears on average.

These ranking-based metrics complement RMSE and MAE
by emphasizing the ordering and relevance of recommendations, which is critical for real-world recommendation systems.
By using both prediction and ranking metrics, we gain a
comprehensive understanding of model performance in terms
of both accuracy and relevance.

B. Baseline Models

The performance of Singular Value Decomposition (SVD)
two traditional matrix factorization
was compared against
techniques: Alternating Least Squares (ALS) and Neural Collaborative Filtering (NCF). These methods serve as strong
baselines for recommendation tasks due to their effectiveness
and computational efficiency.

Alternating Least Squares (ALS) iteratively optimizes user
and item latent factors to minimize prediction error. This
approach is scalable and robust but assumes linear interactions
between users and items, which can limit its ability to capture
complex patterns.

Singular Value Decomposition (SVD), a popular linear
algebra-based method, decomposes the user-item interaction

Fig. 3. Comparison of Models

matrix into latent factors. SVD excels in explaining variance
and uncovering latent relationships in the data. While it also
relies on linear relationships, its ability to reduce dimensionality often results in better performance for capturing user
preferences compared to ALS.

In terms of prediction accuracy, SVD achieved an RMSE
score of 0.9015, outperforming ALS’s RMSE of 0.9229 and
NCF’s RMSE of 0.9502. Both ALS and SVD demonstrate
strong predictive capabilities; however, they share limitations
in modeling non-linear interactions, which are essential for
addressing the sparsity and diversity inherent in large datasets.
NCF, despite its higher RMSE, leverages neural networks to
capture complex relationships, making it a compelling choice
for modern recommendation tasks requiring flexibility.

By highlighting these performance metrics, we underscore
the strengths of SVD in providing accurate recommendations
while acknowledging the evolving landscape of recommendation systems where models like NCF are explored for their
potential advantages in capturing non-linear patterns.

Fig. 4. Ranked Movies

C. Results

The results of the experiments highlight both the strengths
and limitations of the models. The bar chart comparing ALS,
SVD, and NCF showcases their RMSE and MAE scores.

on sufficient training data can make it less effective in environments with sparse datasets. Furthermore, SVD does not
inherently incorporate contextual information, such as movie
metadata or user demographics, which could enhance the
quality of recommendations.

B. Opportunities for Improvement

Several enhancements could be explored to address these
limitations and further improve the recommendation system.
One promising direction is to incorporate additional contextual
information, such as movie metadata (genres, release year,
cast) or user attributes (age, location, preferences). This could
be achieved by extending the SVD architecture into a hybrid
model that combines collaborative filtering with content-based
approaches, allowing for more informed recommendations,
especially for less-rated or newly added movies.

Another avenue for improvement involves addressing data
sparsity. Techniques such as pre-training embedding layers
using matrix factorization methods or employing transfer
learning from other datasets can provide better initialization and improve convergence and performance. Additionally,
introducing regularization techniques like L2 regularization
could help mitigate overfitting, particularly when training on
highly skewed datasets.

Incorporating implicit feedback—such as clicks, views, or
watch duration—could enhance the robustness of the system.
Implicit data is often more abundant
than explicit ratings
and can provide a more comprehensive understanding of
user preferences. Furthermore, experimenting with advanced
techniques like hybrid models or integrating deep learning approaches could improve the model’s ability to capture complex
relationships and contextual information.

Future work should also focus on improving scalability.
Implementing distributed training methods using frameworks
like Apache Spark or TensorFlow Distributed can help manage
the computational demands of large datasets. Exploring realtime recommendation capabilities would allow the system to
adapt dynamically to new data and changing user preferences.
Overall, these enhancements would not only address current
limitations but also position the SVD-based system as a state-of-the-art solution for personalized movie recommendations.

Fig. 5. Recommendation output using SVD

SVD performed the best
in terms of prediction accuracy,
achieving the lowest RMSE (0.9015) and MAE (0.6944). ALS
followed closely with an RMSE of 0.9229 and MAE of 0.7119.
NCF, while showing a higher RMSE of 0.9502, achieved a
comparable MAE of 0.7301. This performance discrepancy
reflects NCF’s focus on ranking and personalization over raw
prediction accuracy.

In addition to prediction metrics, a visualization of the
most popular movies by vote count revealed insights into
user preferences. Titles like The Big Lebowski, Soapdish,
and 2016: Obama’s America received the highest interactions,
indicating a skew in user ratings toward specific well-known or
genre-defining films. This concentration of votes highlights the
impact of user biases on the recommendation process, which
may affect model performance.

While NCF underperformed slightly in prediction-based
metrics like RMSE, its ranking-based metrics (HR, NDCG,
and MRR) outperformed ALS and SVD, demonstrating
its suitability for personalized recommendation tasks. This
strength lies in NCF’s ability to model non-linear interactions,
making it a more flexible framework for diverse datasets.
The popularity distribution analysis further underscores the
importance of addressing biases in the data, which could
be mitigated by integrating contextual information such as
movie genres, release years, or user demographics. Ultimately,
while ALS and SVD remain reliable choices for simpler
recommendation systems, NCF offers a promising approach
for next-generation systems that demand adaptability and
personalization.

VI. IMPROVEMENTS AND FUTURE WORK

VII. CONCLUSION

A. Limitations

While the implemented Singular Value Decomposition
(SVD) model effectively captures latent relationships in useritem interactions,
it has notable limitations. One primary
shortcoming is its assumption of linear relationships, which
may restrict its ability to model complex, non-linear user pref-
erences. This can result in less accurate predictions for certain
user-item interactions compared to more flexible models like
Neural Collaborative Filtering (NCF).

Additionally, SVD may struggle with data sparsity, particularly in scenarios where user ratings are limited. While it
performs well with explicit feedback, the model’s reliance

This study implemented a movie recommendation system
using Singular Value Decomposition (SVD) and compared
its performance against traditional matrix factorization techniques, including Alternating Least Squares (ALS) and Neural Collaborative Filtering (NCF). SVD achieved the lowest
RMSE score of 0.9015, outperforming ALS with an RMSE of
0.9229 and NCF with an RMSE of 0.9502. This demonstrates
SVD’s effectiveness in predicting user ratings while still capturing important latent relationships between users and items.
the analysis of user
and movie rating distributions provided insights into user
preferences and biases, which were effectively modeled by
SVD. While NCF showed strengths in ranking-based metrics

In addition to prediction accuracy,

such as Hit Ratio, NDCG, and MRR, its higher computational
complexity and reliance on extensive training data may limit
its practicality in certain scenarios.

Despite these challenges, SVD’s flexibility and efficiency
make it highly applicable in real-world recommendation systems. Its ability to handle explicit feedback effectively allows
for reliable recommendations based on user ratings. While
issues such as data sparsity remain, the model’s potential for
integration with advanced techniques, such as hybrid models
that combine collaborative filtering with content-based approaches, underscores its relevance in the evolving landscape
of recommender systems.

Overall, SVD represents a robust approach for next generation recommendation systems, balancing predictive accuracy with the ability to provide tailored user experiences.

VIII. REFERENCES

• X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua,
”Neural Collaborative Filtering,” in Proceedings of the
26th International Conference on World Wide Web, 2017,
pp. 173-182.

• F. Xue et al., ”Deep Item-based Collaborative Filtering
for Top-N Recommendation,” ACM Transactions on Information Systems, vol. 37, no. 3, pp. 1-25, 2019.

• M. F. Dacrema, P. Cremonesi, and D. Jannach, ”Are
We Really Making Much Progress? A Worrying Analysis
of Recent Neural Recommendation Approaches,” in Proceedings of the 13th ACM Conference on Recommender
Systems, 2019, pp. 101-109.

• J. Bobadilla, F. Ortega, A. Hernando, and A. Guti´errez,
”Recommender systems survey,” Knowledge-Based Sys-
tems, vol. 46, pp. 109-132, 2013.

• Y. Koren, R. Bell, and C. Volinsky, ”Matrix Factorization
Techniques for Recommender Systems,” Computer, vol.
42, no. 8, pp. 30-37, 2009.
