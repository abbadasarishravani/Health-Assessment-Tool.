import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
#  Load the dataset
# -----------------------------
df = pd.read_csv('dataset.csv')

features = ["Age", "Gender", "SystolicBP", "Cholesterol", "BloodSugar", "Smoking", "Alcohol", "ActivityLevel"]
X = df[features]
y_diabetes = df["DiabetesRisk"]
y_heart = df["HeartRisk"]

# -----------------------------
#  Train ML Models
# -----------------------------
X_train, X_test, y_d_train, y_d_test = train_test_split(X, y_diabetes, test_size=0.2, random_state=42)
_, _, y_h_train, y_h_test = train_test_split(X, y_heart, test_size=0.2, random_state=42)

model_diabetes = LogisticRegression()
model_diabetes.fit(X_train, y_d_train)

model_heart = LogisticRegression()  
model_heart.fit(X_train, y_h_train)

# -----------------------------

def safe_input(prompt, default=0):
    try:
        return float(input(prompt))
    except:
        print(f" Invalid input. Using default = {default}")
        return default

# -----------------------------
#  Collect User Input
# -----------------------------
def get_user_input():
    print("\n Please enter your health details:\n")
    name = input("Your Name: ")
    age = int(safe_input("Age: "))
    gender = int(safe_input("Gender (1=Male, 0=Female): "))
    height = float(safe_input("Height in cm: "))
    weight = float(safe_input("Weight in kg: "))
    systolic = int(safe_input("Systolic BP: "))
    chol = int(safe_input("Cholesterol: "))
    sugar = int(safe_input("Blood Sugar: "))
    smoke = int(safe_input("Do you smoke? (1=Yes, 0=No): "))
    alcohol = int(safe_input("Do you drink alcohol? (1=Yes, 0=No): "))
    activity = int(safe_input("Activity level (0=Low, 1=Moderate, 2=High): "))

    bmi = round(weight / ((height / 100) ** 2), 2)

    user_df = pd.DataFrame([[age, gender, systolic, chol, sugar, smoke, alcohol, activity]], columns=features)
    return name, bmi, user_df, age

# -----------------------------
#  Predict health risks
# -----------------------------
def predict_risks(user_df, age, bmi):
    d_prob = model_diabetes.predict_proba(user_df)[0][1]
    h_prob = model_heart.predict_proba(user_df)[0][1]

    # Adjust probabilities based on age (optional tweak)
    if age > 50:
        d_prob += 0.05
        h_prob += 0.05

    # Risk thresholds
    d_risk = d_prob >= 0.6
    h_risk = h_prob >= 0.5
    o_risk = bmi > 25
    o_prob = min(round((bmi - 18.5) / 10, 2), 1.0) if o_risk else 0.2

    return d_risk, h_risk, o_risk, d_prob, h_prob, o_prob

# -----------------------------
#  Display Personalized Report
# -----------------------------
def show_report(name, bmi, d_risk, h_risk, o_risk, d_prob, h_prob, o_prob):
    print("\n" + "="*55)
    print(f"  Personalized Health Risk Report for {name}")
    print("="*55)
    print(f" BMI: {bmi} {'(Overweight)' if bmi > 25 else '(Normal)' if bmi >= 18.5 else '(Underweight)'}")
    print("\n Risk Levels:")
    print(f"   Diabetes Risk     : {'High ' if d_risk else 'Low '}  | Probability: {d_prob:.2f}")
    print(f"   Heart Disease Risk: {'High ' if h_risk else 'Low '}  | Probability: {h_prob:.2f}")
    print(f"   Obesity Risk      : {'High ' if o_risk else 'Low '}  | BMI-based")

    print("\n Health Recommendations:")
    if d_risk:
        print("  -> Maintain blood sugar levels, reduce processed sugar intake.")
    if h_risk:
        print("  -> Control cholesterol, avoid smoking/alcohol, increase cardio activity.")
    if o_risk:
        print("  -> Balanced diet, increase daily activity, avoid overeating.")
    if not any([d_risk, h_risk, o_risk]):
        print("  => You're currently healthy! Maintain this lifestyle.")
    print("="*55)

# -----------------------------
#  Run the Tool
# -----------------------------
name, bmi, user_df, age = get_user_input()
d_risk, h_risk, o_risk, d_prob, h_prob, o_prob = predict_risks(user_df, age, bmi)
show_report(name, bmi, d_risk, h_risk, o_risk, d_prob, h_prob, o_prob)
