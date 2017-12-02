# The main caller

import os
import read_and_plot
import feature_engineering
import train_model

def read_and_plot_f(data_path, output_path, df):

    read_and_plot.plot_data(df, output_path)

    df = read_and_plot.categorize_income(df, output_path)

    train, test = read_and_plot.create_train_test(df)

    train_c = read_and_plot.copy_data(train)

    read_and_plot.form_scat_plots(train_c, output_path)

    read_and_plot.calc_corrs(train_c, "median_house_value")

    attributes = ["median_house_value",
                "median_income",
                "total_rooms",
                "housing_median_age"]

    read_and_plot.plot_scatter_mat(df, attributes, output_path)


    return train_c, test

def perform_fe(train):

    #train_non_missing = feature_engineering.sort_missing_data(train, ["median_house_value", "ocean_proximity"])
    #op_treated = feature_engineering.convert_to_numeric(train_non_missing)

    #array_one_hot_b = feature_engineering.using_binarizer(op_treated, "housing_cat_encoded")

    #feature_engineering.use_attr_class(op_treated)

    full_tr_array = feature_engineering.using_pipeline(train, "median_house_value")

    return full_tr_array

if __name__ == "__main__":

    data_path = os.path.join(os.getcwd(), "data/housing.csv")
    output_path = os.path.join(os.getcwd(), "chapter_2/outputs")

    target_var = "median_house_value"

    df = read_and_plot.read_data(data_path)
    df = read_and_plot.categorize_income(df, output_path)
    train, test = read_and_plot.create_train_test(df)

    tr_data = perform_fe(train.copy())
    te_data = perform_fe(test.copy())
    labels = train["median_house_value"].values
    te_labels = test["median_house_value"].values

    fitted_model_tree = train_model.fit_treereg_model(tr_data, labels)
    fitted_model_lr = train_model.fit_linreg_model(tr_data, labels)
    fitted_model_rf = train_model.fit_rf_model(tr_data, labels)

    rmse_lr = train_model.evaluate_rmse(fitted_model_lr, tr_data, labels)
    rmse_tree = train_model.evaluate_rmse(fitted_model_tree, tr_data, labels)
    rmse_rf = train_model.evaluate_rmse(fitted_model_rf, tr_data, labels)

    cross_val_rmse, cross_val_rmse_mean, cross_val_rmse_std  = \
        train_model.implement_cross_val_score(fitted_model_tree, tr_data, labels)
    #print(cross_val_rmse_mean, cross_val_rmse_std, rmse_tree)

    cross_val_rmse_lr, cross_val_rmse_mean_lr, cross_val_rmse_std_lr  = \
        train_model.implement_cross_val_score(fitted_model_lr, tr_data, labels)
    #print(cross_val_rmse_mean_lr, cross_val_rmse_std_lr, rmse_lr)

    cross_val_rf, cross_val_rf_mean, cross_val_rf_std  = \
        train_model.implement_cross_val_score(fitted_model_rf, tr_data, labels)
    #print(cross_val_rf_mean, cross_val_rf_std, rmse_rf)

    est = train_model.grid_search(tr_data, labels)
    rand_est = train_model.random_grid_search(tr_data, labels)

    test_rmse = train_model.fit_test(te_data, te_labels, rand_est)
    print(test_rmse)
