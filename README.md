# Abstract
This pet-project addresses the boosting uncertainty problem by using the implementation the virtual ensemble described in this article 
[A. Ustimenko, L. Prokhorenkova and A. Malinin, Uncertainty in Gradient Boosting via Ensembles” (2020), arXiv preprint arXiv:2006.10562](https://arxiv.org/pdf/2006.10562v2.pdf).

As a dataset, time series of product sales is used, where target variable is a number of sales per day.

metrics_validation.py contains the metrics for evaluation the model and class for group time series validation written from scratch

# Description of the problem
In those high-risk tasks where machine learning applied is is crucial to estimate uncertainty in the predictions to adoid mistakes. A virtual ensemble comprised of the sub-models from the one trained gradient boosting model can resolve this problem. ![boosting_image](https://user-images.githubusercontent.com/106237959/194895711-823c91d1-6415-4e31-8330-bd94cd9ec749.png)

By using the virtual ensemble, the knoweldge uncertainty was estimated: \
![image](https://user-images.githubusercontent.com/106237959/194896319-cba50a4c-2a1c-43d2-83fe-2357c2b8728a.png)
