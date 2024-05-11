from flask import Flask, render_template, request
import pandas as pd
from models import load_models

app = Flask(__name__)

# Load models and scaler
models = load_models()
knn_model = models['knn_model']
dt_model = models['dt_model']
lr_model = models['lr_model']
scaler = models['scaler']

# Define the list of columns in the expected sequence
columns_sequence = ['votes', 'rate_transformed', 'Afghan', 'Afghani', 'African', 'American', 'Andhra', 'Arabian', 'Asian', 'Assamese', 'Australian', 'Awadhi', 'BBQ', 'Bar Food', 'Belgian', 'Bengali', 'Beverages', 'Bihari', 'Biryani', 'Bohri', 'British', 'Bubble Tea', 'Burger', 'Burmese', 'Cantonese', 'Charcoal Chicken', 'Chettinad', 'Chinese', 'Coffee', 'Continental', 'Cuisine Bakery', 'Cuisine Cafe', 'Desserts', 'Drinks Only', 'European', 'Fast Food', 'Finger Food', 'French', 'German', 'Goan', 'Greek', 'Grill', 'Gujarati', 'Healthy Food', 'Hot dogs', 'Hyderabadi', 'Ice Cream', 'Indonesian', 'Iranian', 'Italian', 'Japanese', 'Jewish', 'Juices', 'Kashmiri', 'Kebab', 'Kerala', 'Konkan', 'Korean', 'Lebanese', 'Lucknowi', 'Maharashtrian', 'Malaysian', 'Mangalorean', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Mithai', 'Modern Indian', 'Momos', 'Mongolian', 'Mughlai', 'Naga', 'Nepalese', 'North Eastern', 'North Indian', 'Oriya', 'Paan', 'Pan Asian', 'Parsi', 'Pizza', 'Portuguese', 'Rajasthani', 'Raw Meats', 'Roast Chicken', 'Rolls', 'Russian', 'Salad', 'Sandwich', 'Seafood', 'Sindhi', 'Singaporean', 'South American', 'South Indian', 'Spanish', 'Sri Lankan', 'Steak', 'Street Food', 'Sushi', 'Tamil', 'Tea', 'Tex-Mex', 'Thai', 'Tibetan', 'Turkish', 'Vegan', 'Vietnamese', 'Wraps', 'Bar', 'Beverage Shop', 'Bhojanalya', 'Casual Dining', 'Club', 'Confectionery', 'Delivery', 'Dessert Parlor', 'Dhaba', 'Fine Dining', 'Food Court', 'Food Truck', 'Irani Rest Type Cafee', 'Kiosk', 'Lounge', 'Meat Shop', 'Mess', 'Microbrewery', 'Pub', 'Quick Bites', 'Rest Type Bakery', 'Rest Type Cafe', 'Sweet Shop', 'Takeaway', 'Review_Count']

@app.route('/')
def index():
    return render_template('index.html', columns_sequence=columns_sequence)

@app.route("/predict", methods=["POST"])
def predict():
    # Get numerical data from form submission
    votes = float(request.form['votes'])
    rate_transformed = float(request.form['rate_transformed'])
    Review_Count = float(request.form.get('Review_Count', 0))

    # Get categorical data (checkboxes)
    categorical_data = []
    for col in columns_sequence[2:-1]:  # Exclude 'votes', 'rate_transformed', and 'Review_Count'
        value = request.form.get(col, 0)  # Get the value from the form, defaulting to 0
        categorical_data.append(int(value))  # Convert to int and append to the list

    print("Categorical data:", categorical_data)
    print("Form fields received:", request.form)

    # Create a DataFrame from user input
    user_input = [votes, rate_transformed] + categorical_data + [Review_Count]
    print("--" * 20)
    print("Length of columns_sequence:", len(columns_sequence))
    print("Length of user_input:", len(user_input))
    print("--" * 20)
    print("Columns sequence:", columns_sequence)
    print("User input:", user_input)

    if len(columns_sequence) != len(user_input):
        return f"Error: Number of columns in user input ({len(user_input)}) does not match the expected number of columns ({len(columns_sequence)})."

    input_data = pd.DataFrame([user_input], columns=columns_sequence)

    # Get the selected model from the form
    selected_model = request.form.get('model_selection')

    # Make prediction based on the selected model
    if selected_model == 'knn':
        prediction = knn_model.predict(input_data)
    elif selected_model == 'dt':
        prediction = dt_model.predict(input_data)
    elif selected_model == 'lr':
        prediction = lr_model.predict(input_data)
    else:
        return "Invalid model selection"

    # Return prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)