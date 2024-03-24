# LendingClub Loan Automation with Machine Learning

## Project Overview
This project is aimed at automating LendingClub's loan decisions through advanced machine learning techniques. By analyzing loan data, we develop models to classify loans, predict grades, and interest rates, facilitating quicker and more accurate lending decisions.

## Objectives
- To automate LendingClub's loan decision process using machine learning.
- Perform exploratory data analysis (EDA) to understand the dataset's characteristics.
- Apply statistical inference to support model predictions and interpretations.
- Evaluate various machine learning models, focusing on ensemble methods like LightGBM, for predicting loan outcomes.
- Deploy the best performing model to GCP for real-time loan decision-making.

## Dataset
The analysis utilizes a dataset from LendingClub, containing details on loan applications. The dataset provides insights into loan acceptance, grades, and interest rates, forming the basis of our predictive models. [Download the dataset here](https://storage.googleapis.com/335-lending-club/lending-club.zip).

## Analysis Highlights
- EDA revealed key factors influencing loan decisions, such as credit score, income level, and loan amount.
- Statistical tests helped define the target population and form hypotheses regarding loan outcomes.
- LightGBM emerged as the optimal model due to its performance and efficiency after extensive comparison and hyperparameter tuning.

## Key Findings
- Predictive modeling can significantly enhance the accuracy of loan decisions, optimizing LendingClub's lending process.
- LightGBM model, due to its balanced accuracy and efficiency, is recommended for deployment.

## Future Directions
- Further refinement of the models by incorporating more diverse data and addressing class imbalance.
- Exploration of additional features that could influence loan outcomes, enhancing predictive capabilities.
- Continuous monitoring and updating of the deployed model to adapt to changing lending scenarios.

## How to Use
- Clone this repository to access the analysis notebooks and deployment scripts.
- Follow the notebooks for a step-by-step guide on data preprocessing, model training, and evaluation.
- Use the deployment scripts to deploy the model to GCP for real-time predictions.

## Contributions
We welcome contributions, feedback, and suggestions to improve the models and deployment strategies. Feel free to fork the repository or submit pull requests.

## Acknowledgements
Special thanks to LendingClub for providing the dataset and to the team members who contributed to the success of this project.
