# Machine-Unlearning-Notebook
A personal notebook for studying machine unlearning field

Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017, May 1). Membership Inference Attacks Against Machine Learning Models. IEEE Xplore. https://doi.org/10.1109/SP.2017.41

This article discusses the problem of machine learning models leaking information about their training data set and introduces the concept and methodology of membership inference attacks.

Origin: "Machine Learning as a Service" refers to services provided by internet giants where users can upload their own data sets and pay to build models. Users can then make inference calls to these models through APIs.

Data Sources/Components: Various activities and data of individual users, such as purchasing preferences, health data, online and offline transactions, captured photos, voice commands, and travel locations, can be used as training data for machine learning models.

Membership Inference (Definition): It refers to determining whether a particular record was part of the model's training data by making black-box queries to the model, i.e., obtaining only the output of the model given a specific input. The problem is transformed into a classification problem, using an attack model to differentiate the behavior of the target model on training inputs versus unseen inputs.
 
Approach: To build the attack model, the researchers invented a technique called shadow training. By creating multiple shadow models that mimic the behavior of the target model and training them on labeled inputs and outputs, the attack model can be trained.

Findings: Models created using machine learning as a service platforms can leak a significant amount of information about their training data set. Even without knowledge of the target model's training data distribution, membership inference can still be highly accurate.

Potential Solutions: Strategies to mitigate membership inference attacks include limiting the scope of model predictions, reducing the precision of prediction vectors, increasing entropy, or employing regularization techniques.

Tendency: Supervised learning is preferred as the practical training model method.

Data Leakage: Individual inference + membership inference

Individual: Inferring information about the inputs applied by the model based on its outputs.
Membership: Determining whether a given data record belongs to the model's training data set.

Inference: Attackers have query access to the model, and they may have some background knowledge about the target model's training data set.
 
Model Setup: Two commonly used evaluation metrics are accuracy (the proportion of correctly identified data records as members/non-members of the training data set) and recall (the proportion of training data set members correctly identified). These metrics measure the accuracy and success rate of the attack.

Author's Perspective: The problem of overfitting shares similarities with privacy research, as overfitting limits the predictive and generalization capabilities of the model. Regularization techniques like dropout can help mitigate overfitting and enhance privacy in neural networks. Regularization is also used in differentially private machine learning with objective perturbation methods (which can resist to some extent but only operate on the model outputs and may lower prediction accuracy for small values).

Author Advocacy: Membership inference attacks can serve as metrics for quantifying specific model leakage and evaluating the effectiveness of privacy protection techniques in machine learning services.

Environment: Black-box models trained using Google Prediction API and Amazon ML in the cloud.

Key Contribution: Their attack is a generic quantitative method for understanding how machine learning models leak information about their training data set. The shadow training technique trains the attack model to distinguish between members and non-members of the target model's training data set. The attack does not require any prior knowledge about the target model's training data distribution.



Cao, Y., Alexander Fangxiao Yu, Aday, A., Stahl, E., Merwine, J., & Yang, J. (2018). Efficient Repair of Polluted Machine Learning Systems via Causal Unlearning. https://doi.org/10.1145/3196494.3196517


Efficient Repair of Contaminated Machine Learning through their Causal Unlearning System

The primary attack in data contamination involves injecting maliciously crafted training data into the training set, causing the system to learn incorrect models and subsequently misclassify test samples. An intuitive approach is to remove the contaminated data from the training set and retrain the system.

Manual inspection and cleaning of contamination are impractical, hence Karma is introduced, a system proposed by the authors that is designed to effectively repair contaminated machine learning systems. It achieves high accuracy and recall and can automatically detect contamination.

The process involves launching a data contamination attack, which inevitably leaves traces of causality from the contaminated training samples to the contaminated model and then to the misclassified test samples. By searching through different subsets, the system identifies the subset that causes the maximum misclassification. Some samples reported as misclassified are then subjected to human judgment. The system iteratively cleans the system step by step, collecting all the data before the repair. Detecting outliers can also lead to errors and some samples may go undetected, but if they do not cause misclassification, they are considered to have a low level of harm.

The approach employs heuristic algorithms to balance the search space and speed, handling full space search and retraining of the model. It is also applicable to incremental or decremental machine learning.

Detection involves decision trees and K-means.

In practice, if the detection accuracy reaches the threshold set by the administrator, the removal of contaminated data samples is stopped. Alternatively, if no new clusters are found that cause an increase in detection accuracy, the process is halted. Even if a small number of samples remain, they have minimal impact on the overall model because the model itself is robust, especially when they are not in the cluster.

Karma does not lead to overfitting or underfitting and performs well in SVM and Bayesian systems.

Related topics
Du, M., Chen, Z., Liu, C., Oak, R., & Song, D. (2019). Lifelong Anomaly Detection Through Unlearning. https://doi.org/10.1145/3319535.3363226

Not to learn
Kim, B., Kim, H., Kim, K., Kim, S., & Kim, J. (2019). Learning Not to Learn: Training Deep Neural Networks With Biased Data. Openaccess.thecvf.com. https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Learning_Not_to_Learn_Training_Deep_Neural_Networks_With_Biased_CVPR_2019_paper.html



Schelter, S. (n.d.). “Amnesia” -A Selection of Machine Learning Models That Can Forget User Data Very Fast. Retrieved June 15, 2023, from https://www.cidrdb.org/cidr2020/papers/p32-schelter-cidr20.pdf


Motivation: The motivation is not only to delete user data from the database but also from the machine learning model. However, this requires inefficient and expensive retraining of the affected machine learning model, where the training infrastructure needs to access the original training data again and redeploy the retrained model.

Solution: The solution to address the problem of user data forgetting is through "decremental" updates to the trained machine learning model without the need to access the training data again. We provide efficient decremental update procedures that are applicable to four popular machine learning algorithms, along with a single-threaded implementation based on Rust.

Our approach is based on the general idea of training algorithms of machine learning models to retain intermediate results during the model computation, allowing for incremental updates (merging new user data) or decremental updates (deleting user data). In the following sections, we describe in detail three algorithms for different machine learning tasks (recommendation, regression, classification) and explain their efficient incremental/decremental update processes. For each model, we describe the decremental update through the FORGET function, incremental update through the PARTIAL-FIT function, and provide a detailed explanation of how prediction is performed using the PREDICT function.

Recommendation: For example, in movie recommendation, the system keeps track of movies that frequently appear in a user's viewing history. It then ranks these co-occurring movies and forms the basis for subsequent recommendations. Recommendations for specific users are determined by querying for the most similar items to each item (e.g., by calling the PREDICT function in Algorithm 1) and calculating preference estimates through weighted computation of item similarity and corresponding user history.

Regression: Without loss of generality, let's assume that data for a specific user 'i' is captured in the i-th row vector 'Xi' of the input. The model is trained to make predictions based on this data. A common approach is to use a weight vector 'w' to compute the ridge regression model by solving the normal equation w = (XX+λI)-1Xy. The regularization constant λ is typically selected through cross-validation.

Classification (k-nearest): Assigning labels of the k nearest neighbors to unknown observations. It applies a common approximation technique to speed up the search for nearest neighbors, known as Locality-Sensitive Hashing (LSH). The algorithm utilizes indexing on the data to leverage approximate similarity search in high-dimensional spaces. The index consists of multiple hash tables, where observation results that are close in terms of Euclidean distance are likely to end up in the same hash bucket. We utilize random projection as the hash function to compute bucket indices.

Features: The computation tasks are reduced through parallelization, breaking them down into multiple subtasks that can be executed simultaneously on multiple processors, computing cores, or machines to improve efficiency and speed. There is no need to access the original training data during the forgetting process. In such deployments, model training typically requires offline training processes on clusters or the use of powerful machines in separate infrastructures to access training data. However, this method enables a significantly lower operational complexity setup since the model can be updated in-place within the serving system.
