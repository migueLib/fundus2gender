# Local libraries
from e2e_utils.get_df_from_folder import get_df_from_folder
from e2e_utils.logger import set_logger as sl
from e2e_utils.get_consistency_levels import get_consistency_levels
from e2e_utils.plot_barplot import plot_barplot
from e2e_utils.get_distance_matrix import get_distance_matrix
from e2e_utils.plot_heatmap import plot_heatmap
from e2e_utils.get_summary import get_summary

# Built in libraries
import os

# External libraries
import pandas as pd





################################################################################
def main():
    txt = list()
    # Set working directory
    root = os.path.abspath("D:\\Sciebo\\ukbb_fundus_images\\results_by_images")
    logger.info(f"Setting root folder in {root}")

    # Reading data
    df_true = get_df_from_folder(root, "mo_ic_tc_true_label", names=["true"])
    logger.info(f"Reading true labels: {df_true.shape}")

    df_predicted = get_df_from_folder(root, "_predicted_label", nosub=True)
    logger.info(f"Reading predicted labels: {df_predicted.shape}")

    df_paths = get_df_from_folder(root,"_file_paths")
    logger.info(f"Reading file paths: {df_paths.shape}")

    # Set consistency levels
    df_levels = get_consistency_levels(df_predicted, classes=["male", "female"],
                                       logger=logger)
    df_levels_counts = df_levels.value_counts()
    txt.extend([f"{i}\t{j}" for i,j in df_levels_counts.items()])
    logger.info(f"Getting consistency levels")

    # Plot barplot with consistency levels
    plot_barplot(df_levels_counts, logger=logger)
    logger.info(f"Plotting Barplot for consistency levels")

    # Plot barplot with only inconsistent levels
    plot_barplot(df_levels_counts[2:])
    logger.info(f"Plotting barplot for inconsistent levels")

    # Get and plot distance matrix for predicted and predicted + true labels
    df_pred_dis = get_distance_matrix(df_predicted)
    df_predtrue_dis = get_distance_matrix(pd.concat([df_predicted,
                                        df_true.iloc[:, 0]], axis=1))
    logger.info(f"Calculating distance matrix")

    plot_heatmap(df_pred_dis)
    plot_heatmap(df_predtrue_dis)
    logger.info(f"Plotting heatmap")

    # Output summary
    for i in df_predicted:
        summ = get_summary(df_true["true"], df_predicted[i], logger=logger)
        txt.append(summ[-1])
    logger.info("Getting summary")

    # Writing output to outfile
    with open("C:\\Users\\darkg\\Downloads\\output\\6class.txt", "w") as OUT:
        print("\n".join(txt), file=OUT)


if __name__ == '__main__':
    # Set logger
    logger = sl("info")

    # Run script
    main()
