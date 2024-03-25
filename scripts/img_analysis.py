import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from automate_floes import process_floes

def analyze_algo(ift_path, validation_path, land_mask_path, process: bool = True, algorithm_name: str = ''):

    if not os.path.exists('./process_results'):
        os.mkdir('./process_results')

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

    print('done.')

    return pix_params, floe_params


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
    
