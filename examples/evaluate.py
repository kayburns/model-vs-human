from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    # models = ["resnet50_sup", "vits_sup", "vits_sinsup", "resnet50_dino", "vits_dino", "resnet50_moco", "vits_moco", "r3m", "mvp", "vip"]
    # models = ["resnet50_sup", "vits_sup", "vits_sinsup", "resnet50_dino", "vits_dino", "resnet50_moco", "vits_moco"]
    models = ["vits_dinov2"] # "clip"
    # datasets = ["cue-conflict"] # or e.g. ["cue-conflict", "uniform-noise"]
    datasets = ["imagenet_validation", "cue-conflict"] # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = ["shape-bias"] # c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.

def run_shape_bias_analysis():
    # models = ["resnet50_sup", "vits_sup", "vits_sinsup", "resnet50_dino", "vits_dino", "resnet50_moco", "vits_moco", "r3m", "mvp", "vip"]
    models = ["vits_dinov2"] # "clip"
    models = [model.replace("_", "-") for model in models]
    # read csv files from cue-conflict results
    import pandas as pd
    for model_name in models:
        cue_conlict_result_path = f"/iris/u/kayburns/new_arch/model-vs-human/raw-data/cue-conflict/cue-conflict_{model_name}_session-1.csv"
        df = pd.read_csv(cue_conlict_result_path)
        df['texture'] = df['imagename'].str.split('-', expand=True)[1]
        df['texture'] = df['texture'].str.split('.', expand=True)[0]
        # remove trailing digit from texture name
        df['texture'] = df['texture'].str[:-1]
        df_filter = df[df['category'] != df['texture']]
        shape_count = (df['object_response'] == df['category']).sum()
        texture_count = (df['object_response'] == df['texture']).sum()
        shape_bias = shape_count / (shape_count + texture_count)
        model_name = cue_conlict_result_path.split("/")[-1].split("_")[1]
        print(f"Shape bias of {model_name} is {shape_bias}")
        

if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_shape_bias_analysis()
