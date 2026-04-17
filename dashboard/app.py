import json
import streamlit as st
from utils import query_image

# Set Streamlit page configuration
TITLE = "Attributions Inspector"
st.set_page_config(
    page_title=f"{TITLE}",
    page_icon="🖼️",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Main title
# st.title(f"{TITLE}")
st.markdown("<h1 style='text-align: center; color: white;'>Attributions Inspector</h1>", unsafe_allow_html=True)
# st.write(
#     """
#     #### To analyze attributions do the following:
#     - Set the global and individual parameters (apply to all attributions).
#     - Set the individual parameters (apply to single attributions).  
#     - Press submit to visualize each time you change parameters.
#     """
# )

# Load example configurations from JSON
with open("assets/param_maps/param_registry.json", "r", encoding="utf8") as f:
    PARAMS = json.load(f)

attribution_variables = json.load(open("assets/param_maps/attribution_variables.json"))

if PARAMS["attribution_smoother"] not in st.session_state:
    st.session_state.smoother_method = "no_smoothing"

# Empty line for spacing
st.write("")

global_form = st.form(key="form_global")
col1, _, col2 = global_form.columns([2, 0.5, 2])

col1.write("Global Parameters")

# Create a form to enter settings
# Divide columns even more
col1_1, col1_2 = col1.columns([2, 2])

attribution_method = col1_1.selectbox(
    "Attribution Method",
    options=PARAMS["attribution_method"],
    key="attribution_method"
)
date_time = col1_1.selectbox(
    "Date/Time",
    options=PARAMS["date_time"],
    key="date_time"
)
target_variable = col1_1.selectbox(
    "Target Variable",
    options=PARAMS["target_var"],
    key="target_variable",
    help="The variable to attribute the prediction to."
)
target_lat_lon = col1_1.selectbox(
    "Target Latitude/Longitude",
    options=PARAMS["target_lat_lon"],
    key="target_lat_lon"
)
smoother_method = col1_1.selectbox(
    "Smoothing Method",
    options=PARAMS["attribution_smoother"],
    key="smoother",
    help="Method to smooth the attribution map."
)
st.session_state.update(smoother_method=smoother_method)

# m, n and noise level options (depending on smoother)
m = col1_2.selectbox(
    "m",
    options=PARAMS["attribution_smoother"][st.session_state.smoother_method]["m"],
    help="Controls the level of detail in the attribution map.",
    key="m"
)
n = col1_2.selectbox(
    "n",
    options=PARAMS["attribution_smoother"][st.session_state.smoother_method]["n"],
    help="Controls the level of detail in the attribution map.",
    key="n"
)
# I can do slider once we have a proper list with more than just one element
weights_noise_level = col1_2.selectbox(
    "Weights Noise Level",
    options=PARAMS["attribution_smoother"][st.session_state.smoother_method]["weights_noise_level"],
    help="How much noise to add to the weights before computing the attribution.",
    key="weights_noise_level"
)
# I can do slider once we have a proper list with more than just one element
input_noise_level = col1_2.selectbox(
    "Input Noise Level",
    options=PARAMS["attribution_smoother"][st.session_state.smoother_method]["input_noise_level"],
    help="How much noise to add to the input data before computing the attribution.",
    key="input_noise_level"
)

col2_1, col2_2 = col2.columns([2, 2])  

# Attribution 1
col2_1.write("Attribution 1")
model_name_1 = col2_1.selectbox(
    "Model",
    options=PARAMS["model"],
    key="model_name_1"
)
attribution_variable_1 = col2_1.selectbox(
    "Attribution Variable",
    options=attribution_variables[model_name_1],
    key="attribution_variable_1"
)
gradient_accumulation_strategy_1 = col2_1.selectbox(
    "Gradient Accumulation",
    options=PARAMS["gradient_accumulation_strategy"],
    key="gradient_accumulation_strategy_1"
)

# Attribution 2
col2_2.write("Attribution 2")
model_name_2 = col2_2.selectbox(
    "Model",
    options=PARAMS["model"],
    key="model_name_2"
)
attribution_variable_2 = col2_2.selectbox(
    "Attribution Variable",
    options=attribution_variables[model_name_2],
    key="attribution_variable_2"
)
gradient_accumulation_strategy_2 = col2_2.selectbox(
    "Gradient Accumulation",
    options=PARAMS["gradient_accumulation_strategy"],
    key="gradient_accumulation_strategy_2"
)

col2_1.markdown("---")  # Empty line for spacing
col2_2.markdown("---")  # Empty line for spacing

# Attribution 3
col2_1.write("Attribution 3")
model_name_3 = col2_1.selectbox(
    "Model",
    options=PARAMS["model"],
    key="model_name_3",
)
attribution_variable_3 = col2_1.selectbox(
    "Attribution Variable",
    options=attribution_variables[model_name_3],
    key="attribution_variable_3",
)
gradient_accumulation_strategy_3 = col2_1.selectbox(
    "Gradient Accumulation",
    options=PARAMS["gradient_accumulation_strategy"],
    key="gradient_accumulation_strategy_3",
)

# Attribution 4
col2_2.write("Attribution 4")
model_name_4 = col2_2.selectbox(
    "Model",
    options=PARAMS["model"],
    key="model_name_4",
)
attribution_variable_4 = col2_2.selectbox(
    "Attribution Variable",
    options=attribution_variables[model_name_4],
    key="attribution_variable_4",
)
gradient_accumulation_strategy_4 = col2_2.selectbox(
    "Gradient Accumulation",
    options=PARAMS["gradient_accumulation_strategy"],
    key="gradient_accumulation_strategy_4",
)

global_form.form_submit_button("Submit")

param_dict = {
    "date_time": date_time,
    "attribution_method": attribution_method,
    "target_variable": target_variable,
    "target_lat_lon": target_lat_lon,
    "smoother": smoother_method,
    "m": m,
    "n": n,
    "weights_noise_level": weights_noise_level,
    "input_noise_level": input_noise_level,
    "model_name_1": model_name_1,
    "attribution_variable_1": attribution_variable_1,
    "gradient_accumulation_strategy_1": gradient_accumulation_strategy_1,
    "model_name_2": model_name_2,
    "attribution_variable_2": attribution_variable_2,
    "gradient_accumulation_strategy_2": gradient_accumulation_strategy_2,
    "model_name_3": model_name_3,
    "attribution_variable_3": attribution_variable_3,
    "gradient_accumulation_strategy_3": gradient_accumulation_strategy_3,
    "model_name_4": model_name_4,
    "attribution_variable_4": attribution_variable_4,
    "gradient_accumulation_strategy_4": gradient_accumulation_strategy_4,
}

# Placeholder container for result
result_container = st.empty()

# Show the results
with st.spinner("Fetching the result... (may take a couple of seconds)"):

    # st.markdown(f"### {smoother_method} - {attribution_method}")
    # Split into 3 columns
    col1, col2 = st.columns(2)

    # Attribution 1
    col1.write("#### " + param_dict["model_name_1"].lower() + " - " + param_dict["attribution_variable_1"])
    col1.image(query_image(param_dict, "attribution_1"), use_container_width=True)

    # Attribution 2
    col2.write("#### " + param_dict["model_name_2"].lower() + " - " + param_dict["attribution_variable_2"])
    col2.image(query_image(param_dict, "attribution_2"), use_container_width=True)

    # Attribution 3
    col1.write("#### " + param_dict["model_name_3"].lower() + " - " + param_dict["attribution_variable_3"])
    col1.image(query_image(param_dict, "attribution_3"), use_container_width=True)

    # Attribution 4
    col2.write("#### " + param_dict["model_name_4"].lower() + " - " + param_dict["attribution_variable_4"])
    col2.image(query_image(param_dict, "attribution_4"), use_container_width=True)

# Empty lines for layout spacing
st.markdown("</br>", unsafe_allow_html=True)
st.markdown("</br>", unsafe_allow_html=True)