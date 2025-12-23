import os
from sklearn2c import LinearRegressor

# CHOOSE ONE OF THE MODELS
# Parameter linear_reg_config MUST NOT BE CHANGED 
# SINCE IT IS INCLUDED IN C INFERENCE FILES

#line_model_path = os.path.join("regression_models","linear_regressor_line.joblib")
sine_model_path = os.path.join(".","temperature_prediction_lin.joblib")

export_path = os.path.join(".","linear_reg_config")

#linear_regressor = LinearRegressor.load(line_model_path)
linear_regressor = LinearRegressor.load(sine_model_path)

linear_regressor.export(export_path)