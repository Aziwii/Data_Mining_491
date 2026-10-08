import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
import joblib

"""
We will use this script as the entire ML pipeline.
1. Load and clean the data
2. Preprocess 
3. Trains and benchmarks a model
4. Tests the winning model on 2022 data
5. Runs SHAP and saves the results to a folder
"""

def load_and_split_data(filename):
    """Loads cleaned files, creates binary/multiclass targets, and returns

    X_train, X_val, y_train, y_val, X_test_2022, y_test_2022
    """
    df = pd.read_csv(filename)


    #create binary target - evaluate to either 0 = good, very good, excellent | 1 = fair, poor
    df['target_value'] = (df["perceived_mental_health"] == 4.0).astype(int)

    #separate 2018 (train and val pool) from 2022
    df18 = df[df['year'] == 2018].copy()
    df22 = df[df['year'] == 2022].copy()

    #define the features - drop target and year
    cols_to_drop = ['perceived_mental_health', 'year', 'target_value']

    X_2018 = df18.drop(columns=(cols_to_drop)) #features
    y_2018 = df18["target_value"] #target set
    
    X_test_2022 = df22.drop(columns=(cols_to_drop)) #features
    y_test_2022 = df22['target_value'] #target

    #split the data, X/y_train get 80% ||| X/y_val get 20% - - - both of 2018
    X_train, X_val, y_train, y_val = train_test_split(
        X_2018, y_2018, test_size=0.20, random_state=42, stratify=y_2018
    )

    return X_train, X_val, y_train, y_val, X_test_2022, y_test_2022


def build_preprocessor():
    """Creates and returns the ColumnTransformer (OneHot + Scaler).
    
    
    """
    categorical_cols = [
        'province', 
        'smoking_type', 
        'sex', 
        'main_activity'
    ]
    numeric_cols = ['age_group',
       'household_income_group', 'perceived_general_health',
       'perceived_life_stress', 'community_belonging', 'bmi_self_reported',
       'sleep_hours_per_night', 'screen_time_non_workday',
       'physical_activity_hours_7d', 'physical_activity_level',
       'alcohol_frequency']

    # need to scale the numerical cols and need to encode the categorical cols
    #Onehot turns labels into binary yes or nos
    #standard scalar rescales every col to a mean of 0 and a standard dev of 1, this makes it equal 1-5 is not smaller than 0-70
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols), 
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    return preprocessor




def run_model_benchmark(models_dict, preprocessor, X_train, y_train, X_val, y_val):
    """Loops through LogisticRegression, RandomForest, XGBoost, fits each
    inside a Pipeline, and returns a summary DataFrame of validation metrics.
    """

    #create a dictionary to store the trained pipelines
    trained_pipelines = {}

    for name, model in models_dict.items():

        pipe = Pipeline(steps=[
                            ("preprocessor", preprocessor ),
                            ("smote", SMOTE(random_state=42)),
                            (name, model)
                        ])

        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_val)
        pred_proba = pipe.predict_proba(X_val)[:,1]
        accuracy = pipe.score(X_val, y_val)

        print(f"="*50)
        print(f"Running {name}: ")
        print(f"="*50)

        print("Prediction: ",prediction)
        print("Proba_predication: ", pred_proba)
        print("Accuracy: ", accuracy)

        print("Confusion Matrix: \n", confusion_matrix(y_val, prediction))
        print("Classification Report \n", classification_report(y_val, prediction))
        print("Average Precision Score: ", average_precision_score(y_val, pred_proba))
        print("Roc AUC: ", roc_auc_score(y_val, pred_proba))

        #store the pipeline in the dictionary
        trained_pipelines[name] = pipe


    return trained_pipelines



def evaluate_temporal_drift(best_pipeline, X_2022, y_2022):
    """Evaluates the winning 2018 model on the post-COVID 2022 holdout set

    and prints the performance drop (concept drift).
    """

    prediction = best_pipeline.predict(X_2022)
    pred_proba = best_pipeline.predict_proba(X_2022)[:,1]
    accuracy = best_pipeline.score(X_2022, y_2022)

    #evaluations
    print(f"="*50)
    print(f"Running Regression on 2022 set: ")
    print(f"="*50)

    print("Prediction: ",prediction)
    print("Proba_predication: ", pred_proba)
    print("Accuracy: ", accuracy)

    print("Confusion Matrix: \n", confusion_matrix(y_2022, prediction))
    print("Classification Report \n", classification_report(y_2022, prediction))
    print("Average Precision Score: ", average_precision_score(y_2022, pred_proba))
    print("Roc AUC: ", roc_auc_score(y_2022, pred_proba))


def main():
    # 1. Load and split data
    X_train, X_val, y_train, y_val, X_test_2022, y_test_2022 = load_and_split_data("../clean_data/cchs_2018_2022_combined.csv")
    
    # 2. Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # Fit on training data and transform it
    X_train_transformed = preprocessor.fit_transform(X_train)

    print("Preprocessing Check:")
    print(f"Original X_train shape:    {X_train.shape}")
    print(f"Transformed X_train shape: {X_train_transformed.shape}")


    # feature_names = preprocessor.get_feature_names_out()
    # print("\nAll 27 Transformed Features:")
    # for i, name in enumerate(feature_names, 1):
    #     print(f" {i:2d}. {name}")


    # 3. Train & benchmark baseline models on 2018
    boost_scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models_dict = {
        "regression" : LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42 ),
        "random_forest" : RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
        "xgboost" : XGBClassifier(scale_pos_weight=boost_scale_weight, random_state=42, eval_metric="logloss")
    }

    # 4. Evaluate best model on 2022
    trained_pipelines = run_model_benchmark(models_dict, preprocessor, X_train, y_train, X_val, y_val)
    evaluate_temporal_drift(trained_pipelines["regression"], X_test_2022, y_test_2022)

    #save the model and export
    joblib.dump(
        trained_pipelines["regression"],
        "../clean_data/winning_logistic_regression.joblib",
    )
    print(
        "\nSuccessfully saved the training pipeline to ../clean_data/winning_logistic_regression.joblib"
    )

    #to load the model in another file
    # best_pipeline = joblib.load("../clean_data/winning_logistic_regression.joblib")



if __name__ == "__main__":
    main()