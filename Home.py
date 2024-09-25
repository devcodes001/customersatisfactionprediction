import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from  sklearn.metrics import confusion_matrix,auc,roc_curve,classification_report
import seaborn as sns
from PIL import Image
def main():
    st.title("Restaurant Customer Satisfaction Prediction")
    df = pd.read_csv('csp.csv')
    df1 = pd.read_csv('modelreportimb.csv')
    df2 = pd.read_csv('modelreportb.csv')
    tab1, tab2, tab3 ,tab4 = st.tabs(["**Home**", "**Data**","**Prediction**", "**Conclusion**"])
    with tab1:
        img = Image.open('data.jpg')
        st.image(img, width=500)
        st.subheader("About")
        st.markdown('''
         Restaurant Customer Satisfaction Prediction is a project that focuses on identifying the key factors influencing customer satisfaction in restaurants. The goal is to predict whether a customer will be satisfied or dissatisfied based on various attributes. By analyzing these factors, we aim to provide insights that can help restaurants enhance their services and improve the overall dining experience for their customers.
         
         **Features**
         
         CustomerID: Unique identifier for each customer.
        
         Age: Age of the customer.
         
         Gender: Gender of the customer (Male/Female).
         
         Income: Annual income of the customer in USD.
         
         VisitFrequency: How often the customer visits the restaurant (Daily, Weekly,
         Monthly, Rarely).
         
         AverageSpend: Average amount spent by the customer per visit in USD.
         
         PreferredCuisine: The type of cuisine preferred by the customer (Italian, Chinese, Indian, Mexican, American).
         
         TimeOfVisit: The time of day the customer usually visits (Breakfast,Lunch,Dinner).
         
         GroupSize: Number of people in the customer's group during the visit.
         
         DiningOccasion: The occasion for dining (Casual, Business, Celebration).
         
         MealType: Type of meal (Dine-in, Takeaway).
         
         OnlineReservation: Whether the customer made an online reservation (0: No, 1: Yes).
         
         DeliveryOrder: Whether the customer ordered delivery (0: No, 1: Yes).
         
         LoyaltyProgramMember: Whether the customer is a member of the restaurant's * loyalty program (0: No, 1: Yes).
         
         WaitTime: Average wait time for the customer in minutes.
         Satisfaction Ratings
         
         ServiceRating: Customer's rating of the service (1 to 5).
         
         FoodRating: Customer's rating of the food (1 to 5).
         
         AmbianceRating: Customer's rating of the restaurant ambiance (1 to 5).
         
         **Target Variable**

         HighSatisfaction: Binary variable indicating whether the customer is highly satisfied (1) or not (0). Potential Applications Predictive modeling of customer satisfaction. Analyzing factors that drive customer loyalty and satisfaction. Identifying key areas for improvement in service, food, and ambiance. Optimizing marketing strategies to attract and retain satisfied customers.
        ''')
    with tab2:
        st.subheader("About the Dataset")
        st.markdown(
            '''
             dataset provides comprehensive information on customer visits to restaurants, including demographic details, visit-specific metrics, and customer satisfaction ratings. It is designed to facilitate predictive modeling and analytics in the hospitality industry, focusing on factors that drive customer satisfaction.
            '''
        )

        about = st.selectbox(label='select any parameter', options=['head', 'shape', 'columns','describe'], index=None)
        if about == None:
            st.write('Select any to know about the dataset')
        elif about == 'head':
            st.write('Head of the Dataset:')
            st.dataframe(df.head(), use_container_width=True)
        elif about == 'shape':
            st.write('shape of the dataset:', df.shape)
        elif about == 'columns':
            st.write('columns of the airline dataset:')
            st.dataframe(df.columns, use_container_width=True)
        elif about == 'describe':
            st.dataframe(df.describe(),use_container_width=True)
        st.subheader('Model Report')
        mr = st.selectbox(label='select any parameter', options=['Before Sampling', 'After Sampling'], index=None)
        if mr == 'Before Sampling':
            model1 = pd.read_csv('modelreportimb.csv')
            st.dataframe(model1)

            show = st.toggle(label='plot')
            if show:
                plt.figure(figsize=(6, 6))
                sns.barplot(x='Model', y='Accuracy', data=df1, hue='Model', dodge=False)
                plt.xticks(rotation=90)
                st.pyplot(plt)
        elif mr == 'After Sampling':
            model1 = pd.read_csv('modelreportb.csv')
            st.dataframe(model1)
            show = st.toggle(label='plot')
            if show:
                plt.figure(figsize=(6, 6))
                sns.barplot(x='Model', y='Accuracy', data=df2, hue='Model', dodge=False)
                plt.xticks(rotation=90)
                st.pyplot(plt)



    with tab3:
        Age = st.number_input("Age", min_value=0, max_value=120, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        if Gender == "Male":
            Gender = 1
        elif Gender == "Female":
            Gender = 0
        Income = st.number_input("Income", min_value=0, step=1000)
        VisitFrequency = st.selectbox("Visit Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
        if VisitFrequency == "Daily":
            VisitFrequency = 0
        elif VisitFrequency == "Weekly":
            VisitFrequency = 3
        elif VisitFrequency == "Monthly":
            VisitFrequency = 1
        elif VisitFrequency == "Rarely":
            VisitFrequency = 2
        AverageSpend = st.number_input("Average Spend", min_value=0, step=10)
        PreferredCuisine = st.selectbox("Preferred Cuisine", ["Italian", "American", "Chinese", "Indian", "Mexican"])
        if PreferredCuisine == "Italian":
            PreferredCuisine = 3
        elif PreferredCuisine == "American":
            PreferredCuisine = 0
        elif PreferredCuisine == "Chinese":
            PreferredCuisine =1
        elif PreferredCuisine == "Indian":
            PreferredCuisine = 2
        elif PreferredCuisine == "Mexican":
            PreferredCuisine = 4
        TimeOfVisit = st.selectbox("Time Of Visit", ["Breakfast", "Lunch", "Dinner"])
        if TimeOfVisit == 'Breakfast':
            TimeOfVisit = 0
        elif  TimeOfVisit == "Lunch":
            TimeOfVisit = 2
        elif TimeOfVisit == "Dinner":
            TimeOfVisit = 1
        GroupSize = st.number_input("Group Size", min_value=1, step=1)
        DiningOccasion = st.selectbox("Dining Occasion", ["Casual", "Business","Celebration"])
        if DiningOccasion == "Business":
            DiningOccasion = 0
        elif DiningOccasion == "Casual":
            DiningOccasion = 1
        elif DiningOccasion == "Celebration":
            DiningOccasion = 2
        MealType = st.selectbox("Meal Type", ["Dine-in", "Takeaway"])
        if MealType == "Takeaway":
           MealType = 1
        elif MealType == "Dine-in":
            MealType = 0
        OnlineReservation = st.selectbox("Online Reservation", ["Yes", "No"])
        if OnlineReservation == "Yes":
            OnlineReservation = 1
        elif OnlineReservation == "No":
            OnlineReservation = 0
        DeliveryOrder = st.selectbox("Delivery Order", ["Yes", "No"])
        if DeliveryOrder == "Yes":
            DeliveryOrder = 1
        elif DeliveryOrder == "No":
            DeliveryOrder = 0
        LoyaltyProgramMember = st.selectbox("Loyalty Program Member", ["Yes", "No"])
        if LoyaltyProgramMember == "Yes":
            LoyaltyProgramMember = 1
        elif LoyaltyProgramMember == "No":
            LoyaltyProgramMember = 0
        WaitTime = st.number_input("Wait Time (minutes)", min_value=0, step=1)
        ServiceRating = st.slider("Service Rating", min_value=1, max_value=5, step=1)
        FoodRating = st.slider("Food Rating", min_value=1, max_value=5, step=1)
        AmbianceRating = st.slider("Ambiance Rating", min_value=1, max_value=5, step=1)
        features = [Age,Gender,Income, VisitFrequency, AverageSpend, PreferredCuisine,
        TimeOfVisit, GroupSize, DiningOccasion, MealType, OnlineReservation,
        DeliveryOrder, LoyaltyProgramMember,WaitTime,ServiceRating,FoodRating,
        AmbianceRating]
        scaler = pickle.load(open('scaler.sav', 'rb'))
        model = pickle.load(open('model.sav', 'rb'))
        pred = st.button("PREDICT")
        if pred:
            result = model.predict(scaler.transform([features]))
            if result == 0:
                st.write("Not satisfied")
            else:
                st.write("Highly satisfied")
    with tab4:
        st.subheader("Conclusion")
        st.markdown(
            '''
            In the Restaurant Customer Satisfaction Prediction project, we built a model to predict customer satisfaction using various factors. After performing oversampling to address class imbalance, we initially achieved an accuracy of 87% with the XGBoost (XGB) model. To further improve performance, we applied hyperparameter tuning using Optuna, which increased the model’s accuracy to 92%. This improvement demonstrates the importance of both data preprocessing and model optimization. With the final model, restaurants can gain insights into key factors affecting satisfaction and use these predictions to enhance customer experience and retention strategies.
            '''
        )
        f= st.selectbox("Performance measures ", ["Confusion Matrix", "Classification report",'auc-roc Curve'],index=None)
        y_test = pd.read_csv('y.csv')
        y_pred = pd.read_csv('y_pred.csv')
        if f == "Confusion Matrix":
            cm = confusion_matrix(y_test, y_pred)
            st.title("Confusion Matrix Visualization")
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, square=True, linewidths=.5)
            plt.xlabel('Prediction Label')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)
            plt.clf()

        elif f == "Classification report":
            report = classification_report(y_test, y_pred)
            st.title("Classification Report ")
            st.text("Classification Report:")
            st.text(report)
        elif f == "auc-roc Curve":
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            st.title("AUC-ROC Curve Visualization")
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            st.pyplot(plt)
            plt.clf()





main()
