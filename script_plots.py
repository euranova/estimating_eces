from explicalib.experiments.setups.comparison_with_gt.plots import plot_all_upper


BINARY = "binary_v2_3"
CONFIDENCE = "confidence_v2_3"
CLASSWISE = "classwise_v2_3"


def plot_figure_setup_binary(filter_in=None, cut_at=5000):
    plot_all_upper(setup_name=BINARY, ylim=3,filter_in=filter_in, 
                   title="Outcome of the empirical setup in the binary classification quantification setting", 
                   cut_at=cut_at)
    
def plot_figure_setup_confidence(filter_in=None, cut_at=5000):
    plot_all_upper(setup_name=CONFIDENCE, ylim=7,filter_in=filter_in, 
                   title="Outcome of the empirical setup in the confidence classification quantification setting", 
                   cut_at=cut_at)
    
def plot_figure_setup_classwise(filter_in, cut_at=5000):
    plot_all_upper(setup_name=CLASSWISE, ylim=5, filter_in=filter_in, 
                   title="Outcome of the empirical setup in the class-wise classification quantification setting",
                   cut_at=cut_at)
    
def filter_in(report):
    return "'n_bins': 20" not in report["metric_kwargs"] \
            and "'bandwidth': 0.01" not in report["metric_kwargs"] \
            and "'bandwidth': 0.05" not in report["metric_kwargs"]


# Binary plots
plot_figure_setup_binary(filter_in=filter_in)
plot_figure_setup_binary(filter_in=filter_in, cut_at=500)

# Confidence plots
plot_figure_setup_confidence(filter_in=filter_in)
plot_figure_setup_confidence(filter_in=filter_in, cut_at=500)

# Classwise plots
plot_figure_setup_classwise(filter_in=filter_in)
plot_figure_setup_classwise(filter_in=filter_in, cut_at=500)
