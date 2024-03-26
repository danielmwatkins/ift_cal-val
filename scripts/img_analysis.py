import os
import io
import sys
import powerlaw
import json
import numpy as np
import matplotlib.pyplot as plt
from automate_floes import process_floes

def analyze_algo(ift_path, validation_path, land_mask_path, process: bool = True, algorithm_name: str = '', fsd: bool = False):

    if not os.path.exists('./process_results'):
        os.mkdir('./process_results')

    process = False
    fsd = True

    if process:
        print('Processing ' + algorithm_name + ' results:')
        process_floes(ift_path, validation_path, land_mask_path, algorithm_name)

    print(f'Analyzing results for {algorithm_name}...', end='')


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
    hist, bin_edges = np.histogram(centroid_errors_tp, bins=10, density=True)
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
    hist, bin_edges = np.histogram(area_errors_all, bins=200, density=True)
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

    print('done.')

    return pix_params, floe_params


def fsd_for_images(processed_floes: str, dir_name: str):
    # FSD Power Law Distr.
    for case_no, image in processed_floes.items():

        # Suppress printlines from powerlaw package
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        predicted = [floe['floe_area'] for floe in image['fp_floes']] + [floe['ift_floe_area'] for floe in image['ift_to_man'].values()]

        actual = [floe['floe_area'] for floe in image['fn_floes']] + [floe['ift_floe_area'] for floe in image['ift_to_man'].values()]

        actual_results = powerlaw.Fit(actual)
        predicted_results = powerlaw.Fit(predicted)

        man_alpha, ift_alpha = actual_results.power_law.alpha, predicted_results.power_law.alpha

        
        # Enable printlines again
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        try:
            plt.figure(figsize=(5, 5))
            
            fig2 = actual_results.plot_pdf(color='b', linewidth=2, label='Manual FSD')
            actual_results.power_law.plot_pdf(color='b', linestyle='--', ax=fig2, label=f"Manual fit line, alpha = {str(round(man_alpha, 3))}")
            fig2 = predicted_results.plot_pdf(color='r', linewidth=2, ax=fig2, label='IFT FSD')
            predicted_results.power_law.plot_pdf(color='r', linestyle='--', ax=fig2, label=f"IFT fit line, alpha = {str(round(ift_alpha, 3))}")
            
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
    
