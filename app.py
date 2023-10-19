import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/Imthiyaz/Downloads/A&T prediction - Sheet1.csv")
columns_to_remove = [
    "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophil",
    "HCT", "RDW-CV", "MPV", "PDW", "PCT", "RBC", "WBC"
]

dataset = df.drop(columns=columns_to_remove)

X = dataset.iloc[:, :-2].values
y_anemia = dataset.iloc[:, -2].values
y_thrombocytopenia = dataset.iloc[:, -1].values

X_train, X_test, y_train_anemia, y_test_anemia, y_train_thrombocytopenia, y_test_thrombocytopenia = train_test_split(
    X, y_anemia, y_thrombocytopenia, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

LR_c = LogisticRegression()
Knn_c = KNeighborsClassifier(n_neighbors=5)
SVM_c = SVC(kernel="linear", C=1)
RF_c = RandomForestClassifier(n_estimators=75, max_depth=20)
models = [LR_c, RF_c, Knn_c, SVM_c]

accuracy_scores_anemia = {}
accuracy_scores_thrombocytopenia = {}

for model in models:
    model_name = model.__class__.__name__

    model.fit(X_train, y_train_anemia)
    y_pred_anemia = model.predict(X_test)
    accuracy_anemia = accuracy_score(y_test_anemia, y_pred_anemia)
    accuracy_scores_anemia[model_name] = accuracy_anemia

    model.fit(X_train, y_train_thrombocytopenia)
    y_pred_thrombocytopenia = model.predict(X_test)
    accuracy_thrombocytopenia = accuracy_score(y_test_thrombocytopenia, y_pred_thrombocytopenia)
    accuracy_scores_thrombocytopenia[model_name] = accuracy_thrombocytopenia

    print(f"{model_name} Anemia Accuracy: {accuracy_anemia:.2f}")
    print(f"{model_name} Thrombocytopenia Accuracy: {accuracy_thrombocytopenia:.2f}")
    print()
    
average_accuracy = {}
for model_name in accuracy_scores_anemia.keys():
    average_accuracy[model_name] = (accuracy_scores_anemia[model_name] + accuracy_scores_thrombocytopenia[model_name]) / 2
best_model = max(average_accuracy, key=average_accuracy.get)
print(f"Best Fitting Model = '{best_model}' with an average accuracy of {average_accuracy[best_model]:.2f}")
print('__________________________________________________________________________________')
print()

model_names = list(accuracy_scores_anemia.keys())
accuracy_anemia_scores = list(accuracy_scores_anemia.values())
accuracy_thrombocytopenia_scores = list(accuracy_scores_thrombocytopenia.values())

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_anemia_scores, color='skyblue', label='Anemia')
plt.bar(model_names, accuracy_thrombocytopenia_scores, color='lightcoral', label='Thrombocytopenia', alpha=0.5)
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison - Accuracy Scores')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png") 

#Streamlit App
st.title("Anemia & Thrombocytopenia Prediction")

st.subheader("Enter Patient Data:")
gender = st.radio("Gender[0-Male/1-Female]", (0, 1))
hemoglobin = st.number_input("Hemoglobin (g/dL):")
mch = st.number_input("MCH (pg):")
mhch = st.number_input("MHCH:")
mcv = st.number_input("MCV (fL):")
platelet_count = st.number_input("Platelet Count:")

def predict_anemia(gender, hemoglobin, mch, mhch, mcv):
    input_data = [gender, hemoglobin, mch, mhch, mcv, platelet_count]
    anemia_prediction = RF_c.predict(sc.transform([input_data]))
    return anemia_prediction

def predict_thrombocytopenia(gender, hemoglobin, mch, mhch, mcv):
    input_data = [gender, hemoglobin, mch, mhch, mcv, platelet_count]
    thrombocytopenia_prediction = RF_c.predict(sc.transform([input_data]))
    return thrombocytopenia_prediction

if st.button("Predict Anemia"):
    anemia_result = predict_anemia(gender, hemoglobin, mch, mhch, mcv)
    if anemia_result == 0:
        st.write("You don't have anemia.")
    else:
        st.write("You have anemia. Seek medical advice.")

if st.button("Predict Thrombocytopenia"):
    thrombocytopenia_result = predict_thrombocytopenia(gender, hemoglobin, mch, mhch, mcv)
    if thrombocytopenia_result == 0:
        st.write("You don't have thrombocytopenia.")
    else:
        st.write("You have thrombocytopenia. Seek medical advice.")
