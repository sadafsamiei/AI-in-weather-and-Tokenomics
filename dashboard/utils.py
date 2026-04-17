import json

import streamlit as st

# Query image based on selected parameters by user
def query_image(params, image_context):
    # Path where the results are stored
    RESULTS_PATH = "assets/"

    # use results to create a query
    with open("assets/param_maps/param_to_png.json", "r", encoding="utf8") as f:
        results_json = json.load(f)

    # I know it's a mess here, but I didn't find any other way haha
    attr_1_key = str(params.get("date_time")) + "," + params.get("model_name_1") + "," + str(params.get("target_variable")) + "," + str(params.get("target_lat_lon")) + "," + str(params.get("attribution_variable_1")) + "," + str(params.get("m")) + "," + str(params.get("n")) + "," + str(params.get("weights_noise_level")) + "," + str(params.get("input_noise_level")) + "," + params.get("attribution_method") + "," + params.get("gradient_accumulation_strategy_1") + "," + params.get("smoother")
    attr_2_key = str(params.get("date_time")) + "," + params.get("model_name_2") + "," + str(params.get("target_variable")) + "," + str(params.get("target_lat_lon")) + "," + str(params.get("attribution_variable_2")) + "," + str(params.get("m")) + "," + str(params.get("n")) + "," + str(params.get("weights_noise_level")) + "," + str(params.get("input_noise_level")) + "," + params.get("attribution_method") + "," + params.get("gradient_accumulation_strategy_2") + "," + params.get("smoother")
    attr_3_key = str(params.get("date_time")) + "," + params.get("model_name_3") + "," + str(params.get("target_variable")) + "," + str(params.get("target_lat_lon")) + "," + str(params.get("attribution_variable_3")) + "," + str(params.get("m")) + "," + str(params.get("n")) + "," + str(params.get("weights_noise_level")) + "," + str(params.get("input_noise_level")) + "," + params.get("attribution_method") + "," + params.get("gradient_accumulation_strategy_3") + "," + params.get("smoother")
    attr_4_key = str(params.get("date_time")) + "," + params.get("model_name_4") + "," + str(params.get("target_variable")) + "," + str(params.get("target_lat_lon")) + "," + str(params.get("attribution_variable_4")) + "," + str(params.get("m")) + "," + str(params.get("n")) + "," + str(params.get("weights_noise_level")) + "," + str(params.get("input_noise_level")) + "," + params.get("attribution_method") + "," + params.get("gradient_accumulation_strategy_4") + "," + params.get("smoother")

    try:
        attr_1_img_name = results_json.get(attr_1_key, {})
        attr_2_img_name = results_json.get(attr_2_key, {})
        attr_3_img_name = results_json.get(attr_3_key, {})
        attr_4_img_name = results_json.get(attr_4_key, {})

        # I hate using this but, here we are
        match image_context:
            case "attribution_1":
                return f"{RESULTS_PATH}attributions/{attr_1_img_name}"
            case "attribution_2":
                return f"{RESULTS_PATH}attributions/{attr_2_img_name}"
            case "attribution_3":
                return f"{RESULTS_PATH}attributions/{attr_3_img_name}"
            case "attribution_4":
                return f"{RESULTS_PATH}attributions/{attr_4_img_name}"

    except:
        st.error(f"ERROR: image not found!")
        st.stop()