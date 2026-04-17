

import logging
from datetime import datetime, timedelta

from earth2studio.data.utils import fetch_data
from earth2studio.data import GFS
from earth2studio.utils.coords import map_coords

from assets import Files, ensure_files_and_folders, empty_results_folder

logging.basicConfig(
    filename=Files.experiment_log,
    encoding='utf-8',
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("loading functions and models")

from config_loader import (
    load_config, 
    param_grid_from_config, 
    iter_experiments
)
from utils import (
    get_time_array,
    log_params_to_fig_map,
    get_device, 
    count_experiments,
    save_params_registry,
    post_process_params_map,
    save_arrays,
    log_params_to_array_map
)
from attribution_smoother import AttributionSmoother
from visualize import generate_attribution_viz, post_process_attribution
from model_wrapper import WeatherModelWrapper

GENERATE_VIZ = True
ATTRIBUTION = True

if __name__ == "__main__":
    logging.info("setting up experiment")
    ensure_files_and_folders()

    # setup experiment
    device = get_device()
    logging.info(f"found device: {device}")
    data_source = GFS()  
    cfg = load_config()
    param_grid = param_grid_from_config(cfg)  # for experiment
    save_params_registry(param_grid)  # for dashboard

    from config_loader import MODEL_REG
    models = [(name, MODEL_REG[name]) for name in cfg["model"]]
    date_times = cfg.pop("date_time")
    # remove models and datetime from paramgrid
    cfg.pop('model', None)
    param_grid = param_grid_from_config(cfg)  # for experiment
    count_ = count_experiments(param_grid)*len(models)*len(date_times)
    step = 1
    date_time_step = 0
    # flag on the date and re-download x, gt, and re-compute pred, only if new

    # run experiment
    for model_params in models:
        logging.info("loading model from file")
        model_name, RawModelClass = model_params
        # load model and corresponding data
        prognostic = RawModelClass.load_model(RawModelClass.load_default_package())
        model = WeatherModelWrapper(prognostic, device)

        prognostic_ic = prognostic.input_coords()
        output_coords = prognostic.output_coords(prognostic_ic)

        if hasattr(prognostic, "interp_method"):
            interp_to = prognostic_ic
            interp_method = prognostic.interp_method
        else:
            interp_to = None
            interp_method = "nearest"

        for date_time in date_times:
            date_time_step += 1  
            logging.info("fetching data")
            # add next time_step to date_time
            dt_now = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
            next_date_time = dt_now + timedelta(hours=6)
            next_date_time = str(next_date_time)
            
            x, coords = fetch_data(
                source=data_source,
                time=get_time_array(date_time),
                variable=prognostic_ic["variable"],
                lead_time=prognostic_ic["lead_time"],
                device=device,
                interp_to=interp_to,
                interp_method=interp_method,
            )
            x, coords = map_coords(x, coords, prognostic.input_coords())

            gt, next_coords = fetch_data(
                source=data_source,
                time=get_time_array(next_date_time),
                variable=prognostic_ic["variable"],
                lead_time=prognostic_ic["lead_time"],
                device=device,
                interp_to=interp_to,
                interp_method=interp_method,
            )
            gt, _ = map_coords(gt, next_coords, prognostic.input_coords())

            logging.info("generating prediction")
            y = model(x, coords)
            y, _ = map_coords(y, coords, prognostic.input_coords())

            x = x.squeeze(0).squeeze(0).cpu().numpy()
            gt = gt.squeeze(0).squeeze(0).cpu().numpy()
            y = y.squeeze(0).squeeze(0).detach().cpu().numpy()
    
            arrays = {
                "input_state": x,
                "ground_truth": gt, 
                "prediction": y
            }
            save_arrays(
                arrays, date_time_step
            )
            key = f"model={model_name}, time_stamp={date_time}"                
            log_params_to_array_map("x_gt_y", key, str(date_time_step))

            param_iter = 1
            for params_ in iter_experiments(param_grid):
                logging.info(f"{step}/{count_}: {model_name} - {date_time} - {params_.values()}")
                print(f"--- new experiment: {step}/{count_} ---")
                logging.info("extracting params")
                (   
                    # data structures
                    attribution_smoother_params,  # dict
                    attribution_method_params,  # tuple
                    # single entries
                    target_var,  # str
                    target_lat_lon,  # list acting as a tuple. example: [175, 12]
                    grad_accumulation_strategy,  # str
                ) = params_.values()

                # extract params from data structures elements
                smoother_name = attribution_smoother_params["smoother"]
                m = attribution_smoother_params["m"]
                n = attribution_smoother_params["n"]
                weights_noise_level = attribution_smoother_params["weights_noise_level"]
                input_noise_level = attribution_smoother_params["input_noise_level"]
                attribution_name = attribution_method_params[0]
                attribution_method = attribution_method_params[1]

                logging.info("generating attribution")

                # get index of target variable and attribution variables 
                target_var_idx = prognostic_ic["variable"].tolist().index(target_var)
                target_var_lat_lon = (target_var_idx, *target_lat_lon)
                output_variables_names = [str(var) for var in output_coords["variable"].tolist()]
                
                if ATTRIBUTION:
                    # --- generate attribution ---
                    grad_smoother = AttributionSmoother(
                        model=model,
                        weights_noise_level=weights_noise_level,
                        input_noise_level=input_noise_level,
                        m=m,
                        n=n 
                    )
                    attribution_3d = grad_smoother.generate_attribution(
                        array_3d=x, 
                        coords=coords,  
                        target_var_lat_lon=target_var_lat_lon, 
                        explainer=attribution_method, 
                        grad_accumulation_strategy=grad_accumulation_strategy,
                    )
                else:
                    attribution_3d = x

                logging.info("logging stuff")

                # if we actually have different params to log
                key = ",".join(map(str, [
                    date_time, model_name,
                    target_var, target_lat_lon, m, n,
                    weights_noise_level, input_noise_level,
                    attribution_name, grad_accumulation_strategy, smoother_name
                ]))
                # new key for final experiment
                key = f"model={model_name}, smoother={smoother_name}, attribution_method={attribution_name}, time_stamp={date_time}"                
                # TODO: save only if new date
                arrays = {
                        "attribution": attribution_3d,
                    }
                save_arrays(arrays, key)
                log_params_to_array_map("attribution", key, key)

                # --- generate viz and store results ---
                experiment_names = []
                for var in output_variables_names:
                    experiment_name = f"{step}-{var}"
                    key = ",".join(map(str, [
                        date_time, model_name,
                        target_var, target_lat_lon,
                        var, m, n,
                        weights_noise_level, input_noise_level,
                        attribution_name, grad_accumulation_strategy, smoother_name
                    ]))
                    log_params_to_fig_map(key, experiment_name)
                    experiment_names.append(experiment_name)

                if GENERATE_VIZ:
                    logging.info("generating viz")
                    array_3d, cmap, norm = post_process_attribution(
                        attribution_3d, 
                        grad_accumulation_strategy
                        )
                    generate_attribution_viz(
                        attribution_3d,
                        cmap, norm,
                        target_lat_lon,
                        grad_accumulation_strategy,
                        experiment_names
                    )

                logging.info("completed iter")
                param_iter+=1
                step+=1

    post_process_params_map()

    logging.info(f"-- experiment end --")