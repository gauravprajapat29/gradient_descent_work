import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
from sklearn.linear_model import LinearRegression


dataset =pd.read_csv("Salary_Data.csv")
X = dataset["YearsExperience"].ravel()
Y = dataset["Salary"].ravel()
Shape = dataset.shape

def result_data(x,y):
    l = LinearRegression()
    l.fit(x,y)
    return (l.coef_ , l.intercept_)

def result_m(m=0,x=0,y=0,itr=0,no_data=0,lr=0):
    try:
        m = float(m)
        c = 25792.20019866871
        lr = float(lr)
        x = x
        y = y
        no_data = no_data
        for i in range(itr):
            slope_m = (-2 * np.sum((y - m * x - c) * x)) / no_data
            m = m - (slope_m * lr)
        return  (m,c)
    except:
        pass

def result_c(c=0,x=0,y=0,itr=0,no_data=0,lr=0):
    try:
        c = float(c)
        m = 9449.96232146
        lr = float(lr)
        x = x
        y = y
        no_data = no_data
        for i in range(itr):
            slope_c = (-2 * np.sum(y - m * x - c)) / no_data
            c = c - (slope_c * lr)
        return  (m,c)
    except:
        pass

def result_both(m=0,c=0,x=0,y=0,itr=0,no_data=0,lr=0):
    try:
        m = float(m)
        c = float(c)
        lr = float(lr)
        x = x
        y = y
        no_data = no_data
        for i in range(itr):
            slope_m = (-2 * np.sum((y - m * x - c) * x)) / no_data
            m = m - (slope_m * lr)
            slope_c = (-2 * np.sum(y - m * x - c)) / no_data
            c = c - (slope_c * lr)
        return  (m,c)
    except:
        pass


def sklearn_result():
    st.header("Sklearn Find Optimal Solution")
    st.markdown("""Sklearn Python Library to find best fit line and find both m,c best optimal solution""")

    st.subheader("Example")
    st.dataframe(dataset)
    sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
    st.pyplot(plt.gcf())

    st.subheader("Sklearn Result")
    result = result_data(dataset[["YearsExperience"]], dataset["Salary"])
    st.markdown("Best Fit Line :  y = mx + c")
    st.markdown(f"Best m = {result[0][0]}  and  c = {result[1]}")
    st.code("""
    import pandas as pd 
    from sklearn.linear_model import LinearRegression
    dataset = pd.read_csv("Salary_Data.csv")
    l = LinearRegression()
    l.fit(dataset[["YearsExperience"]],dataset["Salary"])
    print(l.coef_,l.intercept_)
    """)
    st.markdown("Best Fit Line : y = l.coef_ * dataset[\"YearsExperience\"] + l.intercept_")
    y = result[0][0] * dataset["YearsExperience"] + result[1]
    sns.lineplot(dataset["YearsExperience"], y, color="red")
    sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
    st.pyplot(plt.gcf())

def find_m():
    st.header("Find Optimal Solution of m")
    st.markdown("""Using Gradient Descent Technique find optimal solution of m but c is given""")

    st.subheader("Example")
    st.dataframe(dataset)
    sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
    st.pyplot(plt.gcf())

    st.subheader("Gradient Descent Technique Result")
    st.code("""
    import pandas as pd 
    from sklearn.linear_model import LinearRegression
    dataset = pd.read_csv("Salary_Data.csv")
    x = dataset["YearsExperience"].ravel()
    y = dataset["Salary"].ravel()
    m = 0
    c = 25792.20019866871
    lr = 0.01
    for i in range(itr):
        slope_m = (-2 * np.sum((y - m * x - c) * x)) / dataset.shape[0]
        m = m - (slope_m * lr))
    y_prd = m * x + c
    """)
    st.markdown("Best Fit Line :  y = mx + c")
    m = st.sidebar.text_input("Starting m Value ")
    lr = st.sidebar.text_input("learning rate")
    itr = st.sidebar.slider("No of Iteration",0,100)
    try:
        ans = result_m(m,X,Y,int(itr),Shape[0],lr)
        st.markdown(f"m = {ans[0]}  and  c = {ans[1]}")
        y = ans[0] * dataset["YearsExperience"] + ans[1]
        sns.lineplot(dataset["YearsExperience"], y, color="red")
        sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
        st.pyplot(plt.gcf())
    except:
        pass

def find_c():
    st.header("Find Optimal Solution of c")
    st.markdown("""Using Gradient Descent Technique find optimal solution of c but m is given""")

    st.subheader("Example")
    st.dataframe(dataset)
    sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
    st.pyplot(plt.gcf())

    st.subheader("Gradient Descent Technique Result")
    st.code("""
    import pandas as pd 
    from sklearn.linear_model import LinearRegression
    dataset = pd.read_csv("Salary_Data.csv")
    x = dataset["YearsExperience"].ravel()
    y = dataset["Salary"].ravel()
    m = 9449.96232146
    c = 0
    lr = 0.01
    for i in range(itr):
        slope_c = (-2 * np.sum(y - m * x - c)) / dataset.shape[0]
        c = c - (slope_c * lr))
    y_prd = m * x + c
    """)
    st.markdown("Best Fit Line :  y = mx + c")
    c = st.sidebar.text_input("Starting c Value ")
    lr = st.sidebar.text_input("learning rate")
    itr = st.sidebar.slider("No of Iteration",0,500)
    try:
        ans = result_c(c,X,Y,int(itr),Shape[0],lr)
        st.markdown(f"m = {ans[0]}  and  c = {ans[1]}")
        y = ans[0] * dataset["YearsExperience"] + ans[1]
        sns.lineplot(dataset["YearsExperience"], y, color="red")
        sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
        st.pyplot(plt.gcf())
    except:
        pass

def find_both():
    st.header("Find Optimal Solution of m, c")
    st.markdown("""Using Gradient Descent Technique find optimal solution of m and c """)

    st.subheader("Example")
    st.dataframe(dataset)
    sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
    st.pyplot(plt.gcf())

    st.subheader("Gradient Descent Technique Result")
    st.code("""
    import pandas as pd 
    from sklearn.linear_model import LinearRegression
    dataset = pd.read_csv("Salary_Data.csv")
    x = dataset["YearsExperience"].ravel()
    y = dataset["Salary"].ravel()
    m = 0
    c = 0
    lr = 0.01
    for i in range(itr):
        slope_c = (-2 * np.sum(y - m * x - c)) / dataset.shape[0]
        c = c - (slope_c * lr))
        slope_m = (-2 * np.sum((y - m * x - c) * x)) / dataset.shape[0]
        m = m - (slope_m * lr))
    y_prd = m * x + c
    """)
    st.markdown("Best Fit Line :  y = mx + c")
    m = st.sidebar.text_input("Starting m Value ")
    c = st.sidebar.text_input("Starting c Value ")
    lr = st.sidebar.text_input("learning rate")
    itr = st.sidebar.slider("No of Iteration",0,1000)
    try:
        ans = result_both(m,c,X,Y,int(itr),Shape[0],lr)
        st.markdown(f"m = {ans[0]}  and  c = {ans[1]}")
        y = ans[0] * dataset["YearsExperience"] + ans[1]
        sns.lineplot(dataset["YearsExperience"], y, color="red")
        sns.scatterplot(dataset["YearsExperience"], dataset["Salary"],color = "blue")
        st.pyplot(plt.gcf())
    except:
        pass


# --------------------------------------------------

st.set_page_config(page_title="Gradient Descent")

with st.sidebar :
    selected = option_menu(
        menu_title="Linear Regression",
        options=["Sklearn Result","Find m","Find c","Find Both m, c"]
    )
if selected == "Sklearn Result":
    sklearn_result()
elif selected == "Find m":
    find_m()
elif selected == "Find c":
    find_c()
elif selected == "Find Both m, c":
    find_both()