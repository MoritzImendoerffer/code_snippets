import numpy as np
import pandas as pd
import re
from functools import reduce

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

# this is some sort of grouping function which sums all values in a group.
def my_grouper_sum(y, key):
    """ Accepts key value vectors and sums all values sharing the same key (e.g. time etc.)
    :param y vector containint the values to sum
    :param key some sort of key (same length as y) which defines the groups

    e.g y = [1, 3 ,5, 6, 7, 8]
        key = [1, 1, 1, 2, 2, 2]
    """
    y_ = np.array(y)
    key_ = np.array(key)
    # extract all unique groups
    unique_groups = np.unique(key_)
    # list for storing the sums
    values = []
    # iterate over all unique group values and append the sum of the groups
    for group in unique_groups:
        values.append(y_[key_ == group].sum())
    return (unique_groups,np.array(values))

def my_grouper_median(y, key):
    """ Accepts key value vectors takes the median all values sharing the
        same key (e.g. time etc.)
    :param y vector containint the values to sum
    :param key some sort of key (same length as y) which defines the groups

    e.g y = [1, 3 ,5, 6, 7, 8]
        key = [1, 1, 1, 2, 2, 2]
    """
    y_ = np.array(y)
    key_ = np.array(key)
    # extract all unique groups
    unique_groups = np.unique(key_)
    # list for storing the sums
    values = []
    # iterate over all unique group values and append the sum of the groups
    for group in unique_groups:
        values.append(np.median(y_[key_ == group]))
    return (unique_groups,np.array(values))

def my_grouper_mean(y, key):
    """ Accepts key value vectors takes the median all values sharing the
        same key (e.g. time etc.)
    :param y vector containint the values to sum
    :param key some sort of key (same length as y) which defines the groups

    e.g y = [1, 3 ,5, 6, 7, 8]
        key = [1, 1, 1, 2, 2, 2]
    """
    y_ = np.array(y)
    key_ = np.array(key)
    # extract all unique groups
    unique_groups = np.unique(key_)
    # list for storing the sums
    values = []
    # iterate over all unique group values and append the sum of the groups
    for group in unique_groups:
        values.append(np.mean(y_[key_ == group]))
    return (unique_groups,np.array(values))

def my_grouper_std(y, key):
    """ Accepts key value vectors takes the median all values sharing the
        same key (e.g. time etc.)
    :param y vector containint the values to sum
    :param key some sort of key (same length as y) which defines the groups

    e.g y = [1, 3 ,5, 6, 7, 8]
        key = [1, 1, 1, 2, 2, 2]
    """
    y_ = np.array(y)
    key_ = np.array(key)
    # extract all unique groups
    unique_groups = np.unique(key_)
    # list for storing the sums
    values = []
    # iterate over all unique group values and append the sum of the groups
    for group in unique_groups:
        values.append(np.std(y_[key_ == group]))
    return (unique_groups,np.array(values))

def my_grouper_minmax(y, key):
    """ Accepts key value vectors return (min,max) tuple sharing the
        same key (e.g. time etc.)
    :param y vector containint the values to sum
    :param key some sort of key (same length as y) which defines the groups

    e.g y = [1, 3 ,5, 6, 7, 8]
        key = [1, 1, 1, 2, 2, 2]
    """
    y_ = np.array(y)
    key_ = np.array(key)
    # extract all unique groups
    unique_groups = np.unique(key_)
    # list for storing the sums
    values = []
    # iterate over all unique group values and append the sum of the groups
    for group in unique_groups:
        values.append((np.min(y_[key_ == group]), np.max(y_[key_ == group]) ))
    return (unique_groups, np.array(values))

def my_grouper_median_max(y, key):
    """ Accepts key value vectors and ... sharing the same key (e.g. time etc.)
    :param y vector containint the values to sum
    :param key some sort of key (same length as y) which defines the groups

    e.g y = [1, 3 ,5, 6, 7, 8]
        key = [1, 1, 1, 2, 2, 2]

    :returns
    """
    y_ = np.array(y)
    key_ = np.array(key)
    # extract all unique groups
    unique_groups = np.unique(key_)
    # list for storing the sums
    values = []
    # iterate over all unique group values and append the sum of the groups
    for group in unique_groups:
        if y_[key_ == group].std() / np.median(y_[key_ == group]) < 0.3:
            values.append(np.median(y_[key_ == group]) )
        else:
            values.append(y_[key_ == group].max())
    return (unique_groups,np.array(values))


def remove_spines(ax):
    """ Removes the right and upper axis from the given axis object

    :param ax    axis object obtained from plt.figure, plt.subplots etc
    """
    try:
        l = len(ax) # check if multiple axes are present
        for a in ax:
            a.spines['right'].set_visible(False)
            a.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            a.yaxis.set_ticks_position('left')
            a.xaxis.set_ticks_position('bottom')
    except: # only one ax is present
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


def merge_data(filename, data_sheet, layout_sheet):
    """ Merge datframe and corresponding layout file
    """

    df = pd.read_excel(filename, data_sheet, index_col=0)
    data = df.unstack().reset_index()
    data.columns = ['column', 'row', 'data']
    layout = pd.read_excel(filename, layout_sheet, skiprows=1, index_col=None)
    merged = pd.merge(layout, data, on = ['row', 'column'])
    return merged


def reg_plot(axis, model, popt, x_data, y_data, alpha=0.95):
    """Plots the confidence and prediction bounds for arbitrary models
       Assumption: residuals are distributed normally
       https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot

       Inputs:
         axis .. axis handle on which the bands will be plotted
         model .. the fitted model f(x, *popt), has to return the fitted y values
         x_data .. independent data points
         y_data .. dependent data_pints
         alpha .. level of confidence

       Output
         returns nothing but plots the confidence and prediction band

    """
    # Statistics
    n = y_data.size                              # number of observations
    m = popt.size                               # number of parameters
    DF = n - m                                  # degrees of freedom
    t = stats.t.ppf(alpha, n - m)               # used for CI and PI bands

    # Estimates of Error in Data/Model
    yhat = model(x_data, *popt)
    resid = y_data - yhat
    chi2 = np.sum((resid/yhat)**2)             # chi-squared; estimates error in data
    chi2_red = chi2/DF                         # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2)/DF)       # standard deviation of the error

    x2 = np.linspace(np.min(x_data), np.max(x_data), 100)
    y2 = model(x2, *popt)
    conf_int =  t * s_err * np.sqrt(1 / n + (x2 - np.mean(x_data))**2 /
                                    np.sum((x_data - np.mean(x_data))**2))
    pred_int = t * s_err * np.sqrt(1 + 1 / n + (x2 - np.mean(x_data))**2 /
                                   np.sum((x_data - np.mean(x_data))**2))

    axis.plot(x_data, y_data, '.', alpha=0.5, color='C1')
    axis.plot(x2, y2, '-', alpha=0.5, color='C2')

    axis.fill_between(x2, y2 + conf_int, y2 - conf_int, alpha=0.5 , color='C1',
                      label='95 % confidence bounds')

    axis.plot(x2, y2 + pred_int, 'k:', alpha=.5)
    axis.plot(x2, y2 - pred_int, 'k:', alpha=.5, label='95 % prediction bounds')
    axis.legend(loc='best')

def get_data_extraction(data_file, layout_file, experiment):
    """ function merges layout and data file for extraction data
    """

    layout = pd.read_excel(layout_file, 'data' , skiprows = 1)
    # drop row of no samples is stored in this well (empty columns > 10)
    #layout.dropna(axis=0, thresh=10, inplace=True)

    xls = pd.ExcelFile(data_file)
    sheets = xls.sheet_names

    '''read in sheet for sheet, and extract wavelength and plate number
    '''
    # looks for the plate number one or two digits
    # behind _ (?<=_) and before _ (?=_)
    re_plate = re.compile(r'(?<=_)(\d{1,2})(?=_)')

    # looks for the plate wavelength exactly 3 digits
    # behind _ (?<=_)
    re_wavelength = re.compile(r'(?<=_)(\d{3})')

    # get all measured wavelengths
    try:
        waves = [int(re_wavelength.search(sheet).group()) for sheet in sheets]
        waves = np.unique(waves)
    except AttributeError:
        print('Error: At least one sheet name contains no valid wavelength information')

    try:
        plates = [int(re_plate.search(sheet).group()) for sheet in sheets]
    except AttributeError:
        print('Error: sheet name contains no valid wavelength information')


    data_all = []
    for sheet in sheets:
        data = xls.parse(sheet, index_col=0)

        # read the plate number
        plate = re_plate.search(sheet).group()
        wavelength = re_wavelength.search(sheet).group()

        data = data.unstack().reset_index()
        data.columns = ['column', 'row', 'data']
        data['wavelength'] = int(wavelength)

        # add plate information
        data['plate'] = int(plate)
        data_all.append(data)



    d1 = pd.concat(data_all)

    # merge all together
    data = pd.merge(layout, d1, on=['plate','row','column'], how='left', copy=False)

    ''' This is a messy hack in order to obtain the shape I want but
    I can clean the function up later on.
    '''
    waves = data.wavelength.unique()

    # loop through all wavelengths in the dataframe
    for index, wave in enumerate(waves):

        if index == 0:
            # copy the dataframe into a new one
            df = data.loc[data.wavelength == wave, :].copy()
            df['data_{}'.format(str(wave))] = df.data
            df.drop('data', axis=1, inplace=True)

            # set a multiindex in order to join afterwards
            df.set_index(['plate', 'row', 'column'], inplace = True)
        else:
            cstring = 'data_{}'.format(str(wave))

            _ = data.loc[data.wavelength == wave, :].copy()
            _[cstring] = _.data
            _.drop('data', axis=1, inplace=True)

            _.set_index(['plate', 'row', 'column'], inplace=True)

            # merge only the new columns into the df
            df = df.join(_.loc[:, cstring], how='outer')

    writer = pd.ExcelWriter('export.xlsx')
    df.to_excel(writer)
    writer.save()
    writer.close()