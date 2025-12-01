from scraper import fetch_data
from model import PricePredictor
import pandas as pd

def test_system():
    print("Testing Scraper...")
    df = fetch_data("Sector 62")
    if df is not None and not df.empty:
        print("Scraper/Synthetic Data Generation Successful")
        print(df.head(2))
    else:
        print("Scraper Failed")
        return

    print("\nTesting Model Training...")
    predictor = PricePredictor()
    metrics = predictor.train(df)
    print(f"Training Successful. Metrics: {metrics}")

    print("\nTesting Prediction...")
    sample_input = {
        'Location': df['Location'].iloc[0],
        'Area_SqFt': 1200,
        'Age_Years': 5,
        'BHK': 2,
        'Market_Rate_SqFt': 5000,
        'Proximity_to_Transit_km': 2.0,
        'School_Rating': 8.0,
        'Walkability_Score': 75,
        'Green_Space_Area': 0.4,
        'Crime_Rate_Index': 20.0,
        'Lift': 1,
        'Security': 1,
        'Park': 0,
        'Gym': 0,
        'Power Backup': 1,
        'Parking': 1
    }
    price = predictor.predict(sample_input)
    print(f"Predicted Price: {price}")
    print("System Verification Complete!")

if __name__ == "__main__":
    test_system()
