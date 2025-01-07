from collections import OrderedDict
from plotting_scripts.plot_error_lines import plot_error_lines

config = "lam"

grey_col = "#C0C0C090"

model_lookup = OrderedDict()
metric_lookup = OrderedDict()
kwargs = {}

if config == "global":
    metric_path = "/home/joel/probwp_paper_eval/global"
    step_length = 6

    model_lookup["det_ms"] = ("GraphCast*", "#00b9e7", ":")
    model_lookup["det_hi"] = ("Graph-FM", "#8981d3", "-.")
    model_lookup["prob_ms"] = ("Graph-EFM (ms)", "#ff6442", "--")
    model_lookup["prob_hi"] = ("Graph-EFM", "#17c7d2", "-")

    model_lookup["gc_swa"] = ("GraphCast*+SWA", "#e75cd9",
        (0, (3, 1, 1, 1, 1, 1)))

    #  model_lookup["gc"] = ("GraphCast", grey_col, ":")
    #  model_lookup["keisler"] = ("KeislerNet", grey_col, "-")
    #  model_lookup["ifs_ens_peval"] = ("IFS-ENS", grey_col, "--")
    #  model_lookup["ngcm_peval"] = ("NeuralGCM", grey_col, "-.")

    model_lookup["optimal"] = ("Calibrated", grey_col, "-.")

    kwargs["print_metrics"]= {
        "z500": (19, 39),
        "2t": (19, 39),
    }

    metric_lookup["rmse"] = "RMSE"
    metric_lookup["crps"] = "CRPS"
    metric_lookup["spskr"] = "Spread/Skill"

elif config == "lam":
    metric_path = "lam_eval"
    step_length = 3
    kwargs["var_names"] = ["pres_0g", "pres_0s", "nlwrs_0", "nswrs_0", "r_2",
        "r_65", "t_2", "t_65", "t_500", "t_850", "u_65", "u_850", "v_65",
        "v_850", "wvint_0", "z_1000", "z_500",]

    model_lookup["det_ms"] = ("GraphCast*", "#00B9E7", ":")
    model_lookup["det_hi"] = ("Graph-FM", "#8981d3", "-.")
    model_lookup["prob_ms"] = ("Graph-EFM (ms)", "#ff6442", "--")
    model_lookup["prob_hi"] = ("Graph-EFM", "#17c7d2", "-")
    model_lookup["diff_hi"] = ("Graph-Diff", "#e75cd9", "-")

    model_lookup["gc_swa"] = ("GraphCast*+SWA", "#e75cd9",
            (0, (3, 1, 1, 1, 1, 1)))

    model_lookup["optimal"] = ("Calibrated", grey_col, "-.")

    kwargs["print_metrics"]= {
        "z_500": (7, 18),
        "wvint_0": (7, 18),
    }

    metric_lookup["rmse"] = "RMSE"
    metric_lookup["crps"] = "CRPS"
    metric_lookup["spskr"] = "Spread/Skill"

elif config == "ens_size_global":
    metric_path = "/home/joel/probwp_paper_eval/ens_size_global"
    step_length = 6

    model_lookup["5"] = ("5", "#e75cd9",
        (0, (3, 1, 1, 1, 1, 1)))
    model_lookup["10"] = ("10", "#00b9e7", ":")
    model_lookup["20"] = ("20", "#8981d3", "-.")
    model_lookup["40"] = ("40", "#ff6442", "--")
    model_lookup["80"] = ("80", "#17c7d2", "-")

    model_lookup["optimal"] = ("Calibrated", grey_col, "-.")

    metric_lookup["rmse"] = "RMSE"
    metric_lookup["crps"] = "CRPS"
    metric_lookup["spskr"] = "Spread/Skill"

    kwargs["print_metrics"]= {
        "z500": (19, 39),
        "2t": (19, 39),
    }

    kwargs["legend_cols"] = 5

elif config == "ens_size_lam":
    metric_path = "/home/joel/probwp_paper_eval/ens_size_lam"
    step_length = 3
    kwargs["var_names"] = ["pres_0g", "pres_0s", "nlwrs_0", "nswrs_0", "r_2",
        "r_65", "t_2", "t_65", "t_500", "t_850", "u_65", "u_850", "v_65",
        "v_850", "wvint_0", "z_1000", "z_500",]

    model_lookup["5"] = ("5", "#e75cd9",
        (0, (3, 1, 1, 1, 1, 1)))
    model_lookup["10"] = ("10", "#00b9e7", ":")
    model_lookup["25"] = ("25", "#8981d3", "-.")
    model_lookup["50"] = ("50", "#ff6442", "--")
    model_lookup["100"] = ("100", "#17c7d2", "-")

    model_lookup["optimal"] = ("Calibrated", grey_col, "-.")

    metric_lookup["rmse"] = "RMSE"
    metric_lookup["crps"] = "CRPS"
    metric_lookup["spskr"] = "Spread/Skill"

    kwargs["print_metrics"]= {
        "z_500": (7, 18),
        "wvint_0": (7, 18),
    }

    kwargs["legend_cols"] = 5

elif config == "propnet_global":
    metric_path = "/home/joel/probwp_paper_eval/propnet_global"
    step_length = 6

    model_lookup["propnet"] = ("Propagation Networks", "#17c7d2", "-")
    model_lookup["inet"] = ("Interaction Networks", "#ff6442", "--")

    metric_lookup["rmse"] = "RMSE"

    kwargs["legend_cols"] = 4

elif config == "propnet_lam":
    metric_path = "/home/joel/probwp_paper_eval/propnet_lam"
    step_length = 3
    kwargs["var_names"] = ["pres_0g", "pres_0s", "nlwrs_0", "nswrs_0", "r_2",
        "r_65", "t_2", "t_65", "t_500", "t_850", "u_65", "u_850", "v_65",
        "v_850", "wvint_0", "z_1000", "z_500",]

    model_lookup["propnet"] = ("Propagation Networks", "#17c7d2", "-")
    model_lookup["inet"] = ("Interaction Networks", "#ff6442", "--")

    metric_lookup["rmse"] = "RMSE"

    kwargs["legend_cols"] = 4

elif config == "latent_map":
    metric_path = "/home/joel/probwp_paper_eval/latent_map_exp"
    step_length = 3
    kwargs["var_names"] = ["pres_0g", "pres_0s", "nlwrs_0", "nswrs_0", "r_2",
        "r_65", "t_2", "t_65", "t_500", "t_850", "u_65", "u_850", "v_65",
        "v_850", "wvint_0", "z_1000", "z_500",]

    model_lookup["latent_map"] = ("Latent map", "#17c7d2", "-")
    model_lookup["static"] = ("Static", "#8981d3", "--")
    model_lookup["optimal"] = ("Calibrated", grey_col, "-.")

    metric_lookup["rmse"] = "RMSE"
    metric_lookup["crps"] = "CRPS"
    metric_lookup["spskr"] = "Spread/Skill"

    kwargs["legend_cols"] = 6
elif config == "global_icp":
    metric_path = "/home/joel/probwp_paper_eval/global_icp"
    step_length = 6

    model_lookup["det_ms"] = ("GraphCast*", "#00b9e7", ":")
    model_lookup["det_hi"] = ("Graph-FM", "#8981d3", "-.")
    model_lookup["prob_ms"] = ("Graph-EFM (ms)", "#ff6442", "--")
    model_lookup["prob_hi"] = ("Graph-EFM", "#17c7d2", "-")

    model_lookup["gc_swa"] = ("GraphCast*+SWA", "#e75cd9",
        (0, (3, 1, 1, 1, 1, 1)))

    model_lookup["gefm_icp"] = ("Graph-EFM+ICP", "green", ":")
    model_lookup["gc_icp"] = ("GraphCast*+ICP", "pink", "-")
    #model_lookup["gfm_icp"] = ("Graph-FM+ICP", "brown", "--")

    model_lookup["optimal"] = ("Calibrated", grey_col, "-.")

    kwargs["print_metrics"]= {
        "z500": (19, 39),
        "2t": (19, 39),
    }

    metric_lookup["rmse"] = "RMSE"
    metric_lookup["crps"] = "CRPS"
    metric_lookup["spskr"] = "Spread/Skill"

elif config == "lam_icp":
    metric_path = "/home/joel/probwp_paper_eval/lam_icp"
    step_length = 3
    kwargs["var_names"] = ["pres_0g", "pres_0s", "nlwrs_0", "nswrs_0", "r_2",
        "r_65", "t_2", "t_65", "t_500", "t_850", "u_65", "u_850", "v_65",
        "v_850", "wvint_0", "z_1000", "z_500",]

    model_lookup["det_ms"] = ("GraphCast*", "#00B9E7", ":")
    model_lookup["det_hi"] = ("Graph-FM", "#8981d3", "-.")
    model_lookup["prob_ms"] = ("Graph-EFM (ms)", "#ff6442", "--")
    model_lookup["prob_hi"] = ("Graph-EFM", "#17c7d2", "-")

    model_lookup["gc_swa"] = ("GraphCast*+SWA", "#e75cd9",
            (0, (3, 1, 1, 1, 1, 1)))

    model_lookup["gefm_icp"] = ("Graph-EFM+ICP", "green", ":")
    model_lookup["gc_icp"] = ("GraphCast*+ICP", "pink", "-")
    model_lookup["gfm_icp"] = ("Graph-FM+ICP", "brown", "--")

    model_lookup["optimal"] = ("Calibrated", grey_col, "-.")

    kwargs["print_metrics"] = {
        "z_500": (7, 18),
        "wvint_0": (7, 18),
    }

    metric_lookup["rmse"] = "RMSE"
    metric_lookup["crps"] = "CRPS"
    metric_lookup["spskr"] = "Spread/Skill"

else:
    # Custom
    metric_path = "/home/joel/probwp_paper_eval/hi4s4l_vs_hi4s3l"
    model_lookup["hi4s4l"] = "4s4l"
    model_lookup["hi4s3l"] = "4s3l"


#metric_lookup["acc"] = "ACC"

plot_error_lines(
    metric_path,
    model_lookup,
    metric_lookup,
    line_width=0.8,
    unitless_metrics=["spskr",],
    step_length=step_length,
    separate_legend=True,
    **kwargs
)
