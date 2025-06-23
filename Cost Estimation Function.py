import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

price_ranges = {
    "Less Damage": (4000, 8000),
    "Moderate Damage": (8000, 15000),
    "Severe Damage": (15000, 35000)
}

def create_damage_price_dataset(num_samples):
    data = []
    
    for _ in range(num_samples):
        severity = np.random.choice(list(price_ranges.keys()))
        min_price, max_price = price_ranges[severity]
        price = np.random.uniform(min_price, max_price)
        
        data.append({
            "Damage Severity": severity,
            "Price": round(price, 2)
        })

    df = pd.DataFrame(data)
    return df

num_samples = 1000
damage_price_dataset = create_damage_price_dataset(num_samples)

label_encoder = LabelEncoder()
damage_price_dataset['Damage Severity Encoded'] = label_encoder.fit_transform(damage_price_dataset['Damage Severity'])

X = damage_price_dataset[['Damage Severity Encoded']]
y = damage_price_dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_damage_severity(model, img_path):
    severities = ["Less Damage", "Moderate Damage", "Severe Damage"]
    predicted_severity = np.random.choice(severities)
    
    return predicted_severity

def predict_price_for_image(model, img_path):
    predicted_severity = predict_damage_severity(model, img_path)
    
    encoded_severity = label_encoder.transform([predicted_severity])[0]
    
    X_new = pd.DataFrame({'Damage Severity Encoded': [encoded_severity]})
    predicted_price = model.predict(X_new)[0]
    
    return predicted_severity, round(predicted_price, 2)

image_path = "/kaggle/input/car-dmage-data-v4/data3a/validation/01-minor/0011.jpeg"

predicted_severity, predicted_price = predict_price_for_image(model, image_path)

print(f"Predicted Severity: {predicted_severity}, Estimated Price: ₹{predicted_price:.2f}")

def display_image_with_prediction(img_path):
    img = image.load_img(img_path)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted Severity: {predicted_severity}\nEstimated Price: ₹{predicted_price:.2f}")
    plt.axis('off')
    plt.show()

display_image_with_prediction(image_path)
