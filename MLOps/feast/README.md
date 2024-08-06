
Feature store in machine learning is the concept to store features in both online and offline stores for model training and serving purposes. Feature store make sure to provide the consistency between the data used for model training and the data used during online serving to models. In other words, it guarantees that you’re serving the same data to models during training and prediction, eliminating training-prediction skew. Feast is one of the open source tools used for feature store.

You can find the theoretical concepts in the feast website itself. Link Here I want to explain the practical implementation aspect of feast. For the same I have created a video explaining each of the below steps used in feast online and offline feature serving to models.

1. Prepare data set and store in parquet format
2. Create an event_timestamp column to the data
3. create a unique ID and add as a column to the data
4. Do feast init and create feature repo structure (to be modified later)
5. create feature_definition.py file inside feature repo and define data source and feature views
6. do feast apply to register and deploy features to offline store
7. load historical features using get_historical_features from offline feature store
8. Train the model using offline features
9. use feast materialize-incremental to load features to online store
10. get online features using get_online_features
11. do the prediction using the data from online feature store.

Please refer the below video for end to end code explanation.


<a href="http://www.youtube.com/watch?feature=player_embedded&v=iZ8R_EUf_pM" target="_blank"><img src="http://img.youtube.com/vi/iZ8R_EUf_pM/0.jpg" 
alt="MLFlow Live Demo" width="560" height="315" border="10" /></a>


References:
https://kedion.medium.com/creating-a-feature-store-with-feast-part-1-37c380223e2f#:~:text=With%20feast%20materialize%2Dincremental%20%2C%20the,of%20the%20most%20recent%20materialization.

https://docs.feast.dev/
