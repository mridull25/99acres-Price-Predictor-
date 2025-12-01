import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class PricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        
    def train(self, df):
        """
        Trains the Random Forest model on the provided DataFrame.
        """
        # Define features
        self.amenities_list = ['Lift', 'Security', 'Park', 'Gym', 'Power Backup', 'Parking']
        self.advanced_factors = ['Proximity_to_Transit_km', 'School_Rating', 'Walkability_Score', 'Green_Space_Area', 'Crime_Rate_Index']
        
        X = df[['Location', 'Area_SqFt', 'Age_Years', 'BHK', 'Market_Rate_SqFt'] + self.amenities_list + self.advanced_factors]
        y = df['Price']
        
        # Preprocessing for categorical data
        categorical_features = ['Location']
        numerical_features = ['Area_SqFt', 'Age_Years', 'BHK', 'Market_Rate_SqFt'] + self.amenities_list + self.advanced_factors
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', 'passthrough', numerical_features)
            ])
            
        # Define the model pipeline
        self.model = Pipeline(steps=[('preprocessor', self.preprocessor),
                                     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
                                     
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {'mae': mae, 'r2': r2}
        
    def predict(self, input_data):
        """
        Predicts price for a single input dictionary.
        input_data: {'Location': str, 'Area_SqFt': float, 'Age_Years': int, 'BHK': int, 'Market_Rate_SqFt': float, 
                     'Proximity_to_Transit_km': float, 'School_Rating': float, ...}
        """
        if self.model is None:
            raise Exception("Model not trained yet!")
            
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)
        return prediction[0]

    def save_model(self, filepath='property_model.joblib'):
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath='property_model.joblib'):
        self.model = joblib.load(filepath)
