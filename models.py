import os
import pickle
from sklearn.preprocessing import MinMaxScaler

def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    knn_model_path = os.path.join(current_dir, 'knn_model.pkl')
    dt_model_path = os.path.join(current_dir, 'dt_model.pkl')
    lr_model_path = os.path.join(current_dir, 'lr_model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')

    with open(knn_model_path, 'rb') as file:
        knn_model = pickle.load(file)
    with open(dt_model_path, 'rb') as file:
        dt_model = pickle.load(file)
    with open(lr_model_path, 'rb') as file:
        lr_model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Update feature_names
    feature_names = ['votes', 'rate_transformed', 'Afghan', 'Afghani', 'African', 'American', 'Andhra', 'Arabian', 'Asian', 'Assamese', 'Australian', 'Awadhi', 'BBQ', 'Bar Food', 'Belgian', 'Bengali', 'Beverages', 'Bihari', 'Biryani', 'Bohri', 'British', 'Bubble Tea', 'Burger', 'Burmese', 'Cantonese', 'Charcoal Chicken', 'Chettinad', 'Chinese', 'Coffee', 'Continental', 'Cuisine Bakery', 'Cuisine Cafe', 'Desserts', 'Drinks Only', 'European', 'Fast Food', 'Finger Food', 'French', 'German', 'Goan', 'Greek', 'Grill', 'Gujarati', 'Healthy Food', 'Hot dogs', 'Hyderabadi', 'Ice Cream', 'Indonesian', 'Iranian', 'Italian', 'Japanese', 'Jewish', 'Juices', 'Kashmiri', 'Kebab', 'Kerala', 'Konkan', 'Korean', 'Lebanese', 'Lucknowi', 'Maharashtrian', 'Malaysian', 'Mangalorean', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Mithai', 'Modern Indian', 'Momos', 'Mongolian', 'Mughlai', 'Naga', 'Nepalese', 'North Eastern', 'North Indian', 'Oriya', 'Paan', 'Pan Asian', 'Parsi', 'Pizza', 'Portuguese', 'Rajasthani', 'Raw Meats', 'Roast Chicken', 'Rolls', 'Russian', 'Salad', 'Sandwich', 'Seafood', 'Sindhi', 'Singaporean', 'South American', 'South Indian', 'Spanish', 'Sri Lankan', 'Steak', 'Street Food', 'Sushi', 'Tamil', 'Tea', 'Tex-Mex', 'Thai', 'Tibetan', 'Turkish', 'Vegan', 'Vietnamese', 'Wraps', 'Bar', 'Beverage Shop', 'Bhojanalya', 'Casual Dining', 'Club', 'Confectionery', 'Delivery', 'Dessert Parlor', 'Dhaba', 'Fine Dining', 'Food Court', 'Food Truck', 'Irani Rest Type Cafee', 'Kiosk', 'Lounge', 'Meat Shop', 'Mess', 'Microbrewery', 'Pub', 'Quick Bites', 'Rest Type Bakery', 'Rest Type Cafe', 'Sweet Shop', 'Takeaway', 'Review_Count']

    with open('feature_names.pkl', 'wb') as file:
        pickle.dump(feature_names, file)

    print("Feature names models:", feature_names)
    print("Number of features models:", len(feature_names))

    return {
        'knn_model': knn_model,
        'dt_model': dt_model,
        'lr_model': lr_model,
        'scaler': scaler
    }