# Car Price Prediction Using Multiple Regression Models

This project is focused on predicting the price of cars based on various features like `months_old`, `power`, and `kms` using multiple regression models. The models include Linear Regression, Polynomial Regression, Support Vector Regression (SVR), Decision Tree Regressor (DTR), Random Forest Regressor (RFR), Ridge Regression, Lasso Regression, and K-Nearest Neighbors Regressor (KNN).

## Libraries Used

- **Pandas**: Data manipulation and analysis.
- **Numpy**: Mathematical operations and array handling.
- **Seaborn**: Visualization of data.
- **Matplotlib**: Plotting graphs.
- **Statsmodels**: Linear regression and OLS model.
- **Sklearn**: Machine learning models and metrics.
pip install numpy pandas seaborn matplotlib statsmodels scikit-learn
cars_data = pd.read_csv('Span_new.csv')
cars_data = cars_data.drop_duplicates()
cars_data = cars_data.dropna()
cars_data = cars_data.drop(['Unnamed: 0', 'ID'], axis=1)
cars_data = cars_data.rename({'age': 'model_year'}, axis=1)
variables_in_study = cars_data[['months_old', 'power', 'kms', 'price']]
scaler = StandardScaler()
scaler.fit(variables_in_study)
variables_in_study = scaler.transform(variables_in_study)
variables_in_study = pd.DataFrame(variables_in_study, columns=['months_old', 'power', 'kms', 'price'])
independent_variables = variables_in_study[['months_old', 'power', 'kms']]
dependent_variable = variables_in_study['price']
def model_evaluation(y_test, predictions):
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, predictions)
    
    MAE_list.append(mae)
    MSE_list.append(mse)
    RMSE_list.append(rmse)
    R_Squared_list.append(r2)
    
    print("Results of sklearn.metrics: \n")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)
# Example with Linear Regression
regression_model = sm.OLS(y_train, x_train)
results = regression_model.fit()
results.summary()

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x_train)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_train)

# Example with SVR
svr = SVR()
svr.fit(x_train, y_train)
predictions = svr.predict(x_test)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
ax1.plot(MAE_list, color='red', label='MAE', marker="o")
ax1.plot(MSE_list, color='blue', label='MSE', marker="o")
ax1.plot(RMSE_list, color='green', label='RMSE', marker="o")
ax2.plot(R_Squared_list, color='green', label='R-Squared', marker="o")

ax1.legend()
ax2.legend()
ax1.set_xticks(ticks=range(len(MAE_list)))
ax2.set_xticks(ticks=range(len(R_Squared_list)))
ax1.set_xticklabels(models_list, rotation=90)
ax2.set_xticklabels(models_list, rotation=90)
ax1.set_xlabel('Regression Models', labelpad=20)
ax2.set_xlabel('Regression Models', labelpad=20)
ax1.set_ylabel('MAE, MSE, RMSE')
ax2.set_ylabel('R-Squared')
plt.show()
print(np.min(RMSE_list))
print(np.argmin(RMSE_list))
print(models_list[np.argmin(RMSE_list)])

print(np.max(R_Squared_list))
print(np.argmax(R_Squared_list))
print(models_list[np.argmax(R_Squared_list)])
