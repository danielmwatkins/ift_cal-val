import os
import sys
import powerlaw
import json
import numpy as np
import matplotlib.pyplot as plt
from automate_floes import process_floes
from scipy.stats import gaussian_kde
import warnings

def analyze_algo(ift_path, 
                validation_path, 
                land_mask_path, 
                process: bool = True, 
                algorithm_name: str = 'predicted', 
                suppress_file_outputs = True,
                fsd: bool = False, 
                threshold_params: dict = None):

    if not os.path.exists('./process_results'):
        os.mkdir('./process_results')

    if process:
        print('Processing ' + algorithm_name + ' results:')
        processed_floes = process_floes(ift_path, validation_path, land_mask_path, algorithm_name, threshold_params=threshold_params, suppress_file_outputs=suppress_file_outputs)
    else:
        try:
            with open(f'process_results/out_{algorithm_name}.json', 'r') as f:
                processed_floes = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Can't find JSON results for {algorithm_name} processing in process_results directory. Try rerunning with processing.")

    print(f"Analyzing {algorithm_name} results...", end='')
    sys.stdout.flush()

    # Get number of t/f p/n floes/pixels in all images
    pix_vals = {"t_pos": 0, "f_pos": 0, "t_neg": 0, "f_neg": 0}
    floe_vals = {"t_pos": 0, "f_pos": 0, "t_neg": 0, "f_neg": 0}


    for _, data in processed_floes.items():

        for k, _ in pix_vals.items():
            pix_vals[k] += data[k + "_pix"]

        for k, _ in floe_vals.items():
            floe_vals[k] += data[k + "_floes"]

    pix_params = calculate_performance_params(pix_vals, object_wise=False)
    floe_params = calculate_performance_params(floe_vals, object_wise=True)

    fsd_plot = None

    if not suppress_file_outputs:
        centroid_errors_all = []
        centroid_errors_tp = []
        for image in processed_floes.values():
            centroid_errors_all.append([x[0]['centroid_distance'] for x in image['intersections'].values()])

            centroid_errors = []
            for ift, man in image['ift_to_man'].items():
                centroid_errors.append(image['intersections'][ift][0]['centroid_distance'])
            centroid_errors_tp.append(centroid_errors)

        centroid_errors_all = [item for sublist in centroid_errors_all for item in sublist]
        centroid_errors_all = np.sort(centroid_errors_all)

        # Calculate the histogram and bin edges for the PDF
        hist, bin_edges = np.histogram(centroid_errors_all, bins=200, density=True)
        pdf = hist / np.sum(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate the cumulative distribution function (CDF)
        cdf = np.cumsum(pdf)

        

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))
        kde = gaussian_kde(centroid_errors_all)

        x = np.linspace(min(centroid_errors_all), max(centroid_errors_all), 1000)

        # Evaluate the PDF at the given points
        pdf_values = kde(x)

        plt.plot(x, pdf_values, label='PDF')
        
        plt.title('Abs. centroid error PDF')
        plt.xlabel('Absolute centroid error distance (px)')
        plt.ylabel('Probability density')

        # Plot the CDF
        plt.subplot(1, 2, 2)
        plt.plot(bin_edges[1:], cdf)
        plt.title('Abs. centroid error CDF')
        plt.suptitle('Figure X: PDF and CDF of centroid error for all identified floe intersections')
        plt.xlabel('Absolute centroid error distance (px)')
        plt.ylabel('Cumulative probability')
        plt.tight_layout()

        dir_name = './out_' + algorithm_name + '/plots'
        if not os.path.exists('./out_' + algorithm_name):
            os.mkdir('./out_' + algorithm_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        plt.savefig(dir_name + '/centroid_error_all_pdf_cdf.png', dpi=300)

        # Now, just for floes identified as TP
        centroid_errors_tp = [item for sublist in centroid_errors_tp for item in sublist]
        centroid_errors_tp = np.sort(centroid_errors_tp)

        # Calculate the histogram and bin edges for the PDF
        bins = int(len(centroid_errors_tp)/8) + 2
        hist, bin_edges = np.histogram(centroid_errors_tp, bins=bins, density=True)
        pdf = hist / np.sum(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate the cumulative distribution function (CDF)
        cdf = np.cumsum(pdf)

        # Plot the PDF
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))

        kde = gaussian_kde(centroid_errors_tp)

        x = np.linspace(min(centroid_errors_tp), max(centroid_errors_tp), 1000)

        # Evaluate the PDF at the given points
        pdf_values = kde(x)

        plt.plot(x, pdf_values, label='PDF')
        plt.title('Abs. centroid error PDF')
        plt.suptitle('Figure X: PDF and CDF of centroid error for floes identified as true positives')
        plt.xlabel('Absolute centroid error distance (px)')
        plt.ylabel('Probability density')

        # Plot the CDF
        plt.subplot(1, 2, 2)
        plt.plot(bin_edges[1:], cdf)
        plt.title('Abs. centroid error CDF')
        plt.xlabel('Absolute centroid error distance (px)')
        plt.ylabel('Cumulative probability')

        plt.savefig(dir_name + '/centroid_error_tp_pdf_cdf.png', dpi=300)


        # Calculation of area percent error plots
        area_errors_all = []
        area_errors_tp = []
        for image in processed_floes.values():
            area_errors_all.append([x[0]['area_percent_difference'] for x in image['intersections'].values() if x[0]['area_percent_difference'] < 5])

            area_errors = []
            for ift, man in image['ift_to_man'].items():
                area_errors.append(image['intersections'][ift][0]['area_percent_difference'])
            area_errors_tp.append(area_errors)

        
        area_errors_all = [item for sublist in area_errors_all for item in sublist]
        area_errors_all = np.sort(area_errors_all)

        # Calculate the histogram and bin edges for the PDF
        bins = int(len(area_errors_all)/8) + 2
        hist, bin_edges = np.histogram(area_errors_all, bins=bins, density=True)
        pdf = hist / np.sum(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate the cumulative distribution function (CDF)
        cdf = np.cumsum(pdf)

        # Plot the PDF
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))
        kde = gaussian_kde(area_errors_all)

        x = np.linspace(min(area_errors_all), max(area_errors_all), 1000)

        # Evaluate the PDF at the given points
        pdf_values = kde(x)

        plt.plot(x, pdf_values, label='PDF')
        plt.title('Area percent error PDF')
        plt.xlabel('Relative difference between real and predicted areas')
        plt.ylabel('Probability density')

        # Plot the CDF
        plt.subplot(1, 2, 2)
        plt.plot(bin_edges[1:], cdf)
        plt.title('Area percent error CDF')
        plt.suptitle('Figure X: PDF and CDF of floe area error for all identified floe intersections')
        plt.xlabel('Relative difference between real and predicted areas')
        plt.ylabel('Cumulative probability')

        plt.tight_layout()

        plt.savefig(dir_name + '/area_error_all_pdf_cdf.png', dpi=300)

        # Now, just for floes identified as TP
        area_errors_tp = [item for sublist in area_errors_tp for item in sublist]
        area_errors_tp = np.sort(area_errors_tp)

        # Calculate the histogram and bin edges for the PDF
        hist, bin_edges = np.histogram(area_errors_tp, bins=10, density=True)
        pdf = hist / np.sum(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate the cumulative distribution function (CDF)
        cdf = np.cumsum(pdf)

        # Plot the PDF
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))
        kde = gaussian_kde(area_errors_tp)

        x = np.linspace(min(area_errors_tp), max(area_errors_tp), 1000)

        # Evaluate the PDF at the given points
        pdf_values = kde(x)

        plt.plot(x, pdf_values, label='PDF')
        plt.title('Area percent error PDF')
        plt.xlabel('Relative difference between real and predicted areas')
        plt.ylabel('Probability density')

        # Plot the CDF
        plt.subplot(1, 2, 2)
        plt.plot(bin_edges[1:], cdf)
        plt.title('Area percent error CDF')
        plt.suptitle('Figure X: PDF and CDF of floe area error for true positive identified floes')
        plt.xlabel('Relative difference between real and predicted areas')
        plt.ylabel('Cumulative probability')

        print(np.std(area_errors_all))
        print(np.std(area_errors_tp))
        print(np.std(centroid_errors_all))
        print(np.std(centroid_errors_tp))

        plt.savefig(dir_name + '/area_error_tp_pdf_cdf.png', dpi=300)

        area_plots(processed_floes, dir_name)

        qualitative_effects(processed_floes, dir_name)


        if fsd:
            fsd_plot = fsd_for_images(processed_floes, dir_name)
            

    print('done.')

    return pix_params, floe_params, fsd_plot


def area_plots(processed_floes: str, dir_name: str):

    pred_areas = []
    real_areas = []
    pred_adj_areas = []
    
    for case, image in processed_floes.items():

        real = []
        pred = []
        pred_adj = []

        for floe in image['ift_to_man'].values():

            pred_area = floe['ift_floe_area']
            pred.append(pred_area)
            real.append(floe['real_floe_area'])
            # Area correction factor
            adj_area = np.pi * (np.sqrt(pred_area / np.pi) + 6)**2
            pred_adj.append(adj_area)

        pred_areas += pred
        real_areas += real
        pred_adj_areas += pred_adj

        try:
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Perform linear regression to get the line of best fit
                m1, b1 = np.polyfit(real, pred, 1)
                m2, b2 = np.polyfit(real, pred_adj, 1)
                x_line = np.array([min(real), max(real)])
                y_line1 = m1 * x_line + b1
                y_line2 = m2 * x_line + b2
                corr_matrix1 = round(np.corrcoef(real, pred)[0, 1], 4)
                corr_matrix2 = round(np.corrcoef(real, pred_adj)[0, 1], 4)

            plt.figure(figsize=(6, 5))

            # Plot the scatter plot and line of best fit
            plt.scatter(real, pred, label='Matched Floes')
            plt.scatter(real, pred_adj, label='Matched Floes, Area Adjusted')
            plt.plot(x_line, y_line1, color='red', label=f"Line of Best Fit: A(x) = {round(m1, 3)}x + {round(b1)}, R = {corr_matrix1}")
            plt.plot(x_line, y_line2, color='green', label=f"Adj. Line of Best Fit: A(x) = {round(m2, 3)}x + {round(b2)}, R = {corr_matrix2}")
            plt.plot(x_line, x_line, '--', color='black', label='Perfect Match: A(x) = x')

            # Add labels and legend
            plt.xlabel('Real floe area (px)')
            plt.ylabel('Predicted floe area (px)')
            plt.title(f"Predicted vs. real floe area  for case {image['case_number']}, {image['satellite'].title()}")
            plt.legend(prop={'size': 7})
            plt.axis([0, max(x_line), 0, max(y_line2)])

            # Show plot
            loc_dir = dir_name + '/' + case
            if not os.path.exists(loc_dir):
                os.mkdir(loc_dir)

            plt.savefig(f"{loc_dir}/{case}_area_comparisons.png", dpi=300)
            plt.close()
            

        except TypeError:
            plt.close()
            continue

    try:
            
        # Perform linear regression to get the line of best fit
        m1, b1 = np.polyfit(real_areas, pred_areas, 1)
        m2, b2 = np.polyfit(real_areas, pred_adj_areas, 1)
        x_line = np.array([min(real_areas), max(real_areas)])
        y_line1 = m1 * x_line + b1
        y_line2 = m2 * x_line + b2
        corr_matrix1 = round(np.corrcoef(real_areas, pred_areas)[0, 1], 4)
        corr_matrix2 = round(np.corrcoef(real_areas, pred_adj_areas)[0, 1], 4)
        

        plt.figure(figsize=(6, 5))

        # Plot the scatter plot and line of best fit
        plt.scatter(real_areas, pred_areas, label='Matched Floes')
        plt.scatter(real_areas, pred_adj_areas, label='Matched Floes, Area Adjusted')
        plt.plot(x_line, y_line1, color='red', label=f"Line of Best Fit: A(x) = {round(m1, 3)}x + {round(b1)}, R = {corr_matrix1}")
        plt.plot(x_line, y_line2, color='green', label=f"Adj. Line of Best Fit: A(x) = {round(m2, 3)}x + {round(b2)}, R = {corr_matrix2}")
        plt.plot(x_line, x_line, '--', color='black', label='Perfect Match: A(x) = x')

        # Add labels and legend
        plt.xlabel('Real floe area (px)')
        plt.ylabel('Predicted floe area (px)')
        plt.title(f"Figure X: Predicted vs. real floe area for all cases")
        plt.legend(prop={'size': 7})
        plt.axis([0, max(x_line), 0, max(y_line2)])

        # Show plot
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        plt.savefig(f"{dir_name}/area_comparisons.png", dpi=300)
        plt.close()

    except TypeError as e:
        plt.close()
        print('Unable to generate predicted area vs. real area plot for algorithm')


def qualitative_effects(processed_floes: str, dir_name: str):

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for _, image in processed_floes.items():
        image['f1_obj'] = 2 * image['t_pos_floes'] / (2 * image['t_pos_floes'] + image['f_pos_floes'] + image['f_neg_floes'])

    landfast_types = ["yes", "no"]
    landfast_f1s = []
    for label in landfast_types:
        landfast_f1s.append(np.mean([x['f1_obj'] for x in processed_floes.values() if x['visible_landfast_ice'] == label]))

    cloud_types = list(set([x['cloud_category'] for x in processed_floes.values()]))
    cloud_f1s = []
    for type in cloud_types:
        cloud_f1s.append(np.mean([x['f1_obj'] for x in processed_floes.values() if x['cloud_category'] == type]))

    artifact_types = ["yes", "no"]
    artifact_f1s = []
    for type in artifact_types:
        artifact_f1s.append(np.mean([x['f1_obj'] for x in processed_floes.values() if x['artifacts'] == type]))

    # floes_types = ["yes", "no"]
    # floes_f1s = []
    # for type in floes_types:
    #     floes_f1s.append(np.mean([x['f1_obj'] for x in processed_floes.values() if x['visible_floes'] == type]))


    _, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs = axs.flatten()

    axs[1].bar(artifact_types, artifact_f1s)
    axs[1].set_xlabel('Artifacts Present?')
    axs[1].set_ylabel('Average image F1 Score')

    axs[0].bar(landfast_types, landfast_f1s)
    axs[0].set_xlabel('Landfast Ice Present?')
    axs[0].set_ylabel('Average image F1 Score')

    axs[2].bar(cloud_types, cloud_f1s)
    axs[2].set_xlabel('Cloud Type Present?')
    axs[2].set_ylabel('Average image F1 Score')

    plt.suptitle('Figure X: Algorithm performance vs. qualitatively assessed image properties')


    # Display the bar chart
    plt.savefig(f"{dir_name}/qualitative.png", dpi=300)
    plt.close()

    f1s = []
    cloud_fracs = []
    for value in processed_floes.values():
        f1s.append(value['f1_obj'])
        cloud_fracs.append(value['cloud_fraction'])

    m1, b1 = np.polyfit(cloud_fracs, f1s, 1)
    x_line = np.array([min(cloud_fracs), max(cloud_fracs)])
    y_line1 = m1 * x_line + b1
    corr_matrix1 = round(np.corrcoef(cloud_fracs, f1s)[0, 1], 4)

    plt.figure(figsize=(6, 5))

    # Plot the scatter plot and line of best fit
    plt.scatter(cloud_fracs, f1s, label='Matched Floes')
    plt.plot(x_line, y_line1, color='red', label=f"Line of Best Fit: F(x) = {round(m1, 3)}x + {round(b1, 3)}, R = {corr_matrix1}")

    # Add labels and legend
    plt.xlabel('Cloud fraction')
    plt.ylabel('F1 Score')
    plt.title(f"Figure X: Image F1 score vs. image cloud fraction")
    plt.legend(prop={'size': 7})

    plt.savefig(f"{dir_name}/cloud_cover_f1.png", dpi=300)
    plt.close()

    return


def fsd_for_images(processed_floes: str, dir_name: str):

    predicted_all = []
    actual_all = []
    tp_all = []

    # FSD Power Law Distr.
    for case_no, image in processed_floes.items():

        predicted = [floe['floe_area'] for floe in image['fp_floes']] + [floe['ift_floe_area'] for floe in image['ift_to_man'].values()]
        predicted_all += predicted

        tp = [floe['ift_floe_area'] for floe in image['ift_to_man'].values()]
        tp_all += tp

        actual = [floe['floe_area'] for floe in image['fn_floes']] + [floe['real_floe_area'] for floe in image['ift_to_man'].values()]
        actual_all += actual


        # Suppress printlines from powerlaw package
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        actual_results = powerlaw.Fit(actual)
        predicted_results = powerlaw.Fit(predicted)
        tp_results = powerlaw.Fit(tp)

        man_alpha, ift_alpha, tp_alpha = actual_results.power_law.alpha, predicted_results.power_law.alpha, tp_results.power_law.alpha

        
        # Enable printlines again
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        try:
            plt.figure(figsize=(5, 5))
            
            fig2 = actual_results.plot_pdf(color='b', linewidth=2, label='Manual FSD')
            actual_results.power_law.plot_pdf(color='b', linestyle='--', ax=fig2, label=f"Manual fit line, alpha = {str(round(man_alpha, 3))}")
            fig2 = predicted_results.plot_pdf(color='r', linewidth=2, ax=fig2, label='IFT FSD, all floes')
            predicted_results.power_law.plot_pdf(color='r', linestyle='--', ax=fig2, label=f"IFT fit line, all floes, alpha = {str(round(ift_alpha, 3))}")

            try:
                fig2 = tp_results.plot_pdf(color='g', linewidth=2, ax=fig2, label='IFT FSD, TP floes')
                tp_results.power_law.plot_pdf(color='g', linestyle='--', ax=fig2, label=f"IFT fit line, TP floes, alpha = {str(round(tp_alpha, 3))}")
            except ValueError:
                pass


            
            plt.title(f"FSD for case {image['case_number']}, {image['satellite']}")
            plt.xlabel('Floe area x')
            plt.ylabel('Probability P(x)')

            loc_dir = dir_name + '/' + case_no
            if not os.path.exists(loc_dir):
                os.mkdir(loc_dir)

            plt.legend(prop={'size': 7})

            plt.savefig(f"{loc_dir}/{case_no}_fsd.png", dpi=300)

            plt.close()

        except ValueError:
            plt.close()
            continue

    # Suppress printlines from powerlaw package
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    actual_results = powerlaw.Fit(actual_all)
    predicted_results = powerlaw.Fit(predicted_all)
    tp_results = powerlaw.Fit(tp_all)

    man_alpha, ift_alpha, tp_alpha = actual_results.power_law.alpha, predicted_results.power_law.alpha, tp_results.power_law.alpha

    # Enable printlines again
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    try:
        fig2 = actual_results.plot_pdf(color='b', linewidth=2, label='Manual FSD')
        actual_results.power_law.plot_pdf(color='b', linestyle='--', ax=fig2, label=f"Manual fit line, alpha = {str(round(man_alpha, 3))}")
        fig2 = predicted_results.plot_pdf(color='r', linewidth=2, ax=fig2, label='IFT FSD, all floes')
        predicted_results.power_law.plot_pdf(color='r', linestyle='--', ax=fig2, label=f"IFT fit line, all floes, alpha = {str(round(ift_alpha, 3))}")

        try:
            fig2 = tp_results.plot_pdf(color='g', linewidth=2, ax=fig2, label='IFT FSD, TP floes')
            tp_results.power_law.plot_pdf(color='g', linestyle='--', ax=fig2, label=f"IFT fit line, TP floes, alpha = {str(round(tp_alpha, 3))}")
        except ValueError:
            pass
        
        plt.title(f"Figure X: FSD for all cases, filtered IFT")
        plt.xlabel('Floe area x')
        plt.ylabel('Probability P(x)')

        plt.text(3, 8, 'Inset Label', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        plt.legend(prop={'size': 7})

        plt.savefig(f"{dir_name}/all_floes_fsd.png", dpi=300)

        plt.close()

        return fig2

    except ValueError:
        plt.close()



def calculate_performance_params(values, object_wise: bool, beta: float = 1.0):
    tp, tn, fp, fn = values['t_pos'], values['t_neg'], values['f_pos'], values['f_neg']

    tpr = (tp + fn) and tp / (tp + fn) or 0 # recall
    tnr = (tn + fp) and tn / (tn + fp) or 0 # not for obia
    ppv = (tp + fp) and tp / (tp + fp) or 0 # precision
    npv = (tn + fn) and tn / (tn + fn) or 0 # not for obia
    
    accuracy = (tp + tn + fp + fn) and (tp + tn) / (tp + tn + fp + fn) or 0 # not for obia
    balanced_accuracy = (tpr + tnr) / 2 # not for obia
    f1 = (ppv + tpr) and (2 * ppv * tpr) / (ppv + tpr) or 0
    fb = (ppv + tpr) and (1 + beta**2) * (ppv * tpr)/((beta**2 * ppv) + tpr) or 0
    fowlkes_mallows = np.sqrt(ppv * tpr)

    fpr, FOR = 1 - tnr, 1 - npv # not for obia
    fdr, fnr = 1 - ppv, 1 - tpr

    mcc = np.sqrt(tpr * tnr * ppv * npv) - np.sqrt(fpr * FOR * fdr * fnr) # not for obia
    lr_pos = fpr and tpr / fpr  or 0 # not for obia
    lr_neg = tnr and fnr / tnr or 0 # not for obia
    dor = lr_neg and lr_pos / lr_neg or 0 # not for obia
    iou = (fp + fn + tp) and tp / (fp + fn + tp) or 0

    if not object_wise:

        return {'Pixel TPR': tpr, 'Pixel TNR': tnr, 'Pixel PPV': ppv, 'Pixel NPV': npv, 'Pixel Accuracy': accuracy, "Pixel Balanced Accuracy": balanced_accuracy, 'Pixel F1': f1,
                'Pixel Fowlkes-Mallows': fowlkes_mallows, 'Pixel MCC': mcc, 'Pixel Pos. Likelihood Ratio': lr_pos, 'Pixel Neg. Likelihood Ratio': lr_neg, 'Pixel Diagnostic Odds Ratio': dor, 
                'Pixel IoU': iou, "Pixel FB": fb}

    return {'Floe TPR': tpr, 'Floe PPV': ppv, 'Floe F1': f1, 'Floe Fowlkes-Mallows': fowlkes_mallows, 'Floe IoU': iou, "Floe FB": fb}
    
