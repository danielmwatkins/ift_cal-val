import os
import io
import sys
import powerlaw
import json
import numpy as np
import matplotlib.pyplot as plt
from automate_floes import process_floes
import warnings

def analyze_algo(ift_path, 
                validation_path, 
                land_mask_path, 
                process: bool = True, 
                algorithm_name: str = '', 
                fsd: bool = False):

    if not os.path.exists('./process_results'):
        os.mkdir('./process_results')

    process = False
    fsd = True

    if process:
        print('Processing ' + algorithm_name + ' results:')
        process_floes(ift_path, validation_path, land_mask_path, algorithm_name)


    try:
        with open(f'process_results/out_{algorithm_name}.json', 'r') as f:
            processed_floes = json.load(f)
    except FileNotFoundError:
        print(f"Can't find results for {algorithm_name} processing. Try rerunning with processing.")

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

    # Plot the PDF
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))
    plt.title('Abs. centroid error PDF')
    plt.xlabel('Value (px)')
    plt.ylabel('Probability')

    # Plot the CDF
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], cdf, marker='o')
    plt.title('Abs. centroid error CDF')
    plt.xlabel('Value (px)')
    plt.ylabel('Cumulative Probability')
    plt.tight_layout()

    dir_name = './out_' + algorithm_name + '/plots'
    if not os.path.exists('./out_' + algorithm_name):
        os.mkdir('./out_' + algorithm_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    plt.savefig(dir_name + '/centroid_error_all_pdf_cdf.png')

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
    plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))
    plt.title('Abs. centroid error PDF')
    plt.xlabel('Value (px)')
    plt.ylabel('Probability')

    # Plot the CDF
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], cdf, marker='o')
    plt.title('Abs. centroid error CDF')
    plt.xlabel('Value (px)')
    plt.ylabel('Cumulative Probability')

    plt.savefig(dir_name + '/centroid_error_tp_pdf_cdf.png')


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
    bins = int(len(centroid_errors_all)/8) + 2
    hist, bin_edges = np.histogram(area_errors_all, bins=bins, density=True)
    pdf = hist / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(pdf)

    # Plot the PDF
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))
    plt.title('Area percent error PDF')
    plt.xlabel('Value')
    plt.ylabel('Probability')

    # Plot the CDF
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], cdf, marker='o')
    plt.title('Area percent error CDF')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')

    plt.tight_layout()

    plt.savefig(dir_name + '/area_error_all_pdf_cdf.png')


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
    plt.bar(bin_centers, pdf, width=0.5*(bin_centers[1]-bin_centers[0]))
    plt.title('Area percent error PDF')
    plt.xlabel('Value')
    plt.ylabel('Probability')

    # Plot the CDF
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], cdf, marker='o')
    plt.title('Area percent error CDF')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')

    plt.savefig(dir_name + '/area_error_tp_pdf_cdf.png')

    if fsd:
        fsd_for_images(processed_floes, dir_name)

    area_plots(processed_floes, dir_name)

    print('done.')

    return pix_params, floe_params


def area_plots(processed_floes: str, dir_name: str):

    pred_areas = []
    real_areas = []
    
    for case, image in processed_floes.items():

        real = []
        pred = []

        for floe in image['ift_to_man'].values():

            pred.append(floe['ift_floe_area'])
            real.append(floe['real_floe_area'])

        pred_areas += pred
        real_areas += real

        try:
            
            with warnings.catch_warnings(action="ignore"):
                # Perform linear regression to get the line of best fit
                m, b = np.polyfit(real, pred, 1)
                x_line = np.array([min(real), max(real)])
                y_line = m * x_line + b

            plt.figure(figsize=(5, 5))

            # Plot the scatter plot and line of best fit
            plt.scatter(real, pred, label='Matched Floes')
            plt.plot(x_line, y_line, color='red', label=f"Line of Best Fit: A(x) = {round(m, 3)}x + {round(b)}")
            plt.plot(x_line, x_line, '--', color='black', label='Perfect Match: A(x) = x')

            # Add labels and legend
            plt.xlabel('Real floe area (px)')
            plt.ylabel('Predicted floe area (px)')
            plt.title(f"Predicted vs. real floe area  for case {image['case_number']}, {image['satellite'].title()}")
            plt.legend()

            # Show plot
            loc_dir = dir_name + '/' + case
            if not os.path.exists(loc_dir):
                os.mkdir(loc_dir)

            plt.savefig(f"{loc_dir}/{case}_area_comparisons.png")

        except TypeError:
            continue

    try:
            
        # Perform linear regression to get the line of best fit
        m, b = np.polyfit(real_areas, pred_areas, 1)
        x_line = np.array([min(real_areas), max(real_areas)])
        y_line = m * x_line + b


        plt.figure(figsize=(5, 5))

        # Plot the scatter plot and line of best fit
        plt.scatter(real_areas, pred_areas, label='Matched Floes')
        plt.plot(x_line, y_line, color='red', label=f"Line of Best Fit: A(x) = {round(m, 3)}x + {round(b)}")
        plt.plot(x_line, x_line, '--', color='black', label='Perfect Match: A(x) = x')

        # Add labels and legend
        plt.xlabel('Real floe area (px)')
        plt.ylabel('Predicted floe area (px)')
        plt.title(f"Predicted vs. real floe area for all cases")
        plt.legend()

        # Show plot
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        plt.savefig(f"{dir_name}/area_comparisons.png")

    except TypeError as e:
        print(e)
        print('Unable to generate predicted area vs. real area plot for algorithm')


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

            plt.text(3, 8, 'Inset Label', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

            loc_dir = dir_name + '/' + case_no
            if not os.path.exists(loc_dir):
                os.mkdir(loc_dir)

            plt.legend()

            plt.savefig(f"{loc_dir}/{case_no}_fsd.png")

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
        
        plt.title(f"FSD for all cases")
        plt.xlabel('Floe area x')
        plt.ylabel('Probability P(x)')

        plt.text(3, 8, 'Inset Label', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        plt.legend()

        plt.savefig(f"{dir_name}/all_floes_fsd.png")

        plt.close()

    except ValueError:
        plt.close()



def calculate_performance_params(values, object_wise: bool):
    tp, tn, fp, fn = values['t_pos'], values['t_neg'], values['f_pos'], values['f_neg']

    tpr = (tp + fn) and tp / (tp + fn) or 0
    tnr = (tn + fp) and tn / (tn + fp) or 0 # not for obia
    ppv = (tp + fp) and tp / (tp + fp) or 0
    npv = (tn + fn) and tn / (tn + fn) or 0 # not for obia
    
    accuracy = (tp + tn + fp + fn) and (tp + tn) / (tp + tn + fp + fn) or 0 # not for obia
    balanced_accuracy = (tpr + tnr) / 2 # not for obia
    f1 = (ppv + tpr) and (2 * ppv * tpr) / (ppv + tpr) or 0
    fowlkes_mallows = np.sqrt(ppv * tpr)

    fpr, FOR = 1 - tnr, 1 - npv # not for obia
    fdr, fnr = 1 - ppv, 1 - tpr

    mcc = np.sqrt(tpr * tnr * ppv * npv) - np.sqrt(fpr * FOR * fdr * fnr) # not for obia
    lr_pos = fpr and tpr / fpr  or 0 # not for obia
    lr_neg = tnr and fnr / tnr or 0 # not for obia
    dor = lr_neg and lr_pos / lr_neg or 0 # not for obia
    iou = (fp + fn + tp) and tp / (fp + fn + tp) or 0

    if not object_wise:

        return {'tpr': tpr, 'tnr': tnr, 'ppv': ppv, 'npv': npv, 'acc': accuracy, "bal_acc": balanced_accuracy, 'f1': f1,
                'fowlkes-mallows': fowlkes_mallows, 'mcc': mcc, 'lr_pos': lr_pos, 'lr_neg': lr_neg, 'dor': dor, 
                'iou': iou}

    return {'tpr': tpr, 'ppv': ppv, 'f1': f1, 'fowlkes-mallows': fowlkes_mallows, 'iou': iou}
    
