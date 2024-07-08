from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,5)
plt.style.use('fivethirtyeight')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def helloworld():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/', methods=['POST'])
def predict():

    # Load the dataset
    data = pd.read_csv('mental-illnesses-prevalence.csv')
    data = data.round()
    print(data.head())

    # Defining input features (X) and output target (y)
    X = data.drop('Depressive', axis=1)  # Input features
    y = data['Depressive']  # Output target

    # Defining the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Schizophrenia disorders', 'Anxiety', 'Bipolar disorders','Eating disorders']),
        ])

    # Create preprocessing and modeling pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    pipeline.fit(X_train, y_train)

    # Making predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Evaluating the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f'Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}')
    print(f'Train R^2: {r2_train:.2f}, Test R^2: {r2_test:.2f}')

    # Function to predict charges based on user input
    def predict_charges(Schizophrenia_disorders, Anxiety, Bipolar_disorders, Eating_disorders):
        input_data = pd.DataFrame({
            'Schizophrenia disorders': [Schizophrenia_disorders],
            'Anxiety': [Anxiety],
            'Bipolar disorders': [Bipolar_disorders],
            'Eating disorders': [Eating_disorders]
        })
        prediction = pipeline.predict(input_data)
        return prediction[0]


    if request.method == 'POST':
        # Retrieve the text from the textarea
        Schizophrenia_disorders = request.form.get('Schizophrenia')
        Anxiety = request.form.get('Anxiety')
        Bipolar_disorders = request.form.get('Bipolar')
        Eating_disorders = request.form.get('Eating')

    if(Schizophrenia_disorders=="High"):
        Schizophrenia_disorders = 0
    elif(Schizophrenia_disorders=="Low"):
        Schizophrenia_disorders = 0
    else:
        Schizophrenia_disorders = 0

    if(Anxiety=="High"):
        Anxiety = 9
    elif(Anxiety=="Low"):
        Anxiety = 2
    else:
        Anxiety = 4

    if(Bipolar_disorders=="High"):
        Bipolar_disorders = 2
    elif(Bipolar_disorders=="Low"):
        Bipolar_disorders = 0
    else:
        Bipolar_disorders = 1

    if(Eating_disorders=="High"):
        Eating_disorders = 1
    elif(Eating_disorders=="Low"):
        Eating_disorders = 0
    else:
        Eating_disorders = 0

    predicted_charges = predict_charges(Schizophrenia_disorders, Anxiety, Bipolar_disorders, Eating_disorders)
    sug = ""
    if(round(predicted_charges)<=3):
        b = "Low"
        sug = "Regular Exercise: Engage in daily physical activities like walking, yoga, or light jogging to boost mood. Healthy Diet: Maintain a balanced diet rich in fruits, vegetables, and lean proteins. Sleep Hygiene: Ensure consistent sleep patterns with 7-9 hours of sleep each night. Social Interaction: Stay connected with friends and family to prevent isolation. Mindfulness and Relaxation: Practice mindfulness, meditation, or deep breathing exercises."
    elif (round(predicted_charges) <= 5 and round(predicted_charges)>=3):
        b  = "Medium"
        sug = "Therapy: Consider seeing a therapist for cognitive-behavioral therapy (CBT) or other counseling methods. Medication: Consult a healthcare provider about the possibility of antidepressants or other medications. Structured Routine: Create and stick to a daily schedule to provide structure and reduce stress. Support Groups: Join a support group to share experiences and gain insights from others facing similar challenges. Hobbies and Interests: Engage in activities that bring joy and fulfillment, such as reading, gardening, or painting."
    else:
        b = "High"
        sug = "Professional Help: Seek immediate professional help from a psychiatrist or clinical psychologist. Medication Management: Work closely with a healthcare provider to manage medications and monitor their effects. Intensive Therapy: Participate in intensive outpatient programs or inpatient treatment if necessary. Crisis Plan: Develop a crisis plan including emergency contacts and coping strategies. Close Monitoring: Ensure close monitoring by family, friends, or caregivers to provide support and ensure safety."

    a = f'Depression Level : {round(predicted_charges)} The Level is : {b}'

    return render_template('result.html', my_string=a, sug_string=sug)

if __name__ == '__main__':
    app.run(port=3000, debug=True)