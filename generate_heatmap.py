import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import image
from scipy.spatial.distance import cdist
from PIL import Image
from tqdm import tqdm
import argparse

def aggregate_gaze_points(gaze_data_list, N=7):
    if len(gaze_data_list) <= N:
        return gaze_data_list
    else:
        while len(gaze_data_list) > N:
            gaze_point_list = [(item[0], item[1]) for item in gaze_data_list]
            # calculate all distances between two sets of points
            dists = cdist(gaze_point_list, gaze_point_list)
            # the self distance is 0 -> we don't want this so make it large
            dists[dists == 0] = dists.max()
            # get index of smallest distance
            (idx1, idx2) = np.unravel_index(dists.argmin(), dists.shape)
            gaze_data_list[idx1] = (int(0.5*(gaze_data_list[idx1][0]+gaze_data_list[idx2][0])), 
                                    int(0.5*(gaze_data_list[idx1][1]+gaze_data_list[idx2][1])), 
                                    int(0.5*(gaze_data_list[idx1][2]+gaze_data_list[idx2][2])))
            gaze_data_list.remove(gaze_data_list[idx2])
        return gaze_data_list

def filter_gaze(gaze_data_list, t=350):
    gaze_data_filtered = []
    for item in gaze_data_list:
        if item[-1] <350:
            continue
        else:
            gaze_data_filtered.append(item)
    return gaze_data_filtered


def gaussian(x, sx, y=None, sy=None):
    """Returns an array of np arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution
    arguments
    x		-- width in pixels
    sx		-- width standard deviation
    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M

def normalize_map(s_map):
    norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
    return norm_s_map

def draw_heatmap(gazepoints, dispsize, imagesize, startsize, originalsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None, gaussianwhy=200, gaussiansdy=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.
    arguments
    gazepoints		-	a list of gazepoint tuples (x, y)
    
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)
    keyword arguments
    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)
    returns
    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gwhy = gaussianwhy
    gsdwh = gwh // 6 if (gaussiansd is None) else gaussiansd
    gsdwhy = gwhy //6 if (gaussiansdy is None) else gaussiansdy
    gaus = gaussian(gwh, gsdwh, gwhy, gsdwhy)
    # matrix of zeroes
    strt = gwh // 2
    strty = gwhy // 2
    heatmapsize = dispsize[1] + 2 * strty, dispsize[0] + 2 * strt
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strty + gazepoints[i][1] - int(gwhy / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwhy]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwhy - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwhy, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # draw heatmap on top of image
    img_cut = heatmap[startsize[1]:(startsize[1]+imagesize[1]), startsize[0]:(startsize[0]+imagesize[0])]
    # normalize the salience map
    img_cut = normalize_map(img_cut)
    maps = Image.fromarray(img_cut*255).convert('L').resize((originalsize))


    if imagefile:
        img = plt.imread(imagefile)
        plt.imshow(img)
        plt.imshow(maps, alpha=0.6, cmap='jet')
        plt.axis('off')
        plt.savefig(savefilename, bbox_inches='tight', pad_inches = 0)
        plt.close()
    else:
        maps.save(savefilename)


def plot_gazemap(params, image_name, image_id, display_width, display_height, image_x, image_y, start_x, start_y, o_x, o_y, gaze_data, ngaussian=500, sd=75.0, ngaussiany=500, sdy=75.0, alpha=0.5):
    
    ## find the path to original image. If you want to have an overlay image, please give the image_file to the "imagefile" in the draw_heatmap function.
    all_images= os.listdir(os.path.join(params.CUB_dir_path, 'images'))
    for item in all_images:
        if item[4:].lower() in image_name.lower():
            image_file = os.path.join(params.CUB_dir_path, 'images', item, image_name+'.jpg')
            break

    ## visualize all fixation in one heatmap
    if not params.single_fixation:
        if not os.path.exists(os.path.join(params.CUB_GHA_save_path, 'All_Fixation')):
            os.makedirs(os.path.join(params.CUB_GHA_save_path, 'All_Fixation'))
        output_name = os.path.join(params.CUB_GHA_save_path, 'All_Fixation', "%s.jpg"%image_id)
        draw_heatmap(gaze_data, (display_width, display_height), (image_x, image_y), (start_x, start_y), (o_x, o_y), savefilename=output_name, imagefile=None, gaussianwh=ngaussian, gaussiansd=sd, gaussianwhy=ngaussiany, gaussiansdy=sdy)  

    ## visualize single fixation
    else:
        if not os.path.exists(os.path.join(params.CUB_GHA_save_path, 'Single_Fixation', image_id)):
            os.makedirs(os.path.join(params.CUB_GHA_save_path, 'Single_Fixation', image_id))
        for i, gaze in enumerate(gaze_data):
            output_name = os.path.join(params.CUB_GHA_save_path, 'Single_Fixation', image_id, str(i)+".jpg")
            draw_heatmap([gaze], (display_width, display_height), (image_x, image_y), (start_x, start_y), (o_x, o_y), savefilename=output_name, imagefile=None, gaussianwh=ngaussian, gaussiansd=sd, gaussianwhy=ngaussiany, gaussiansdy=sdy)    

def parse_args():
    parser = argparse.ArgumentParser(description='Main script')
    parser.add_argument('--CUB_dir_path', default="./CUB_200_2011", help='path to CUB_200_2011')
    parser.add_argument('--CUB_GHA_save_path', default="./CUB_GHA", help='path to save CUB_GHA')
    parser.add_argument('--gaze_file_path', default="./Fixation.txt", help='path to the fixation file')
    parser.add_argument('--single_fixation', default=False, action='store_true', help='set the flag if you want to plot each fixation seperately')
    return parser.parse_args()



if __name__=='__main__':
    params = parse_args()

    index2image = {}
    f = open(os.path.join(params.CUB_dir_path, 'images.txt'), "r")
    for item in f.readlines():
        item_list = item.split(" ")
        image_name = item_list[1].strip("\n").split("/")[-1]
        index2image[item_list[0]] = image_name.split(".")[0]
    f.close()
    file = open(params.gaze_file_path, 'r') 
    lines = file.readlines()

    for line in tqdm(lines):
        line = line.strip().split(',')
        image_id = line[0]
        gaze_data = []
        if (len(line)-7)%3 != 0:
            print("Some information is missing in image: %s" %image_id)
            exit()
        else:
            for j in range((len(line)-7)//3-1):
                gaze_data.append([int(line[7+3*j]), int(line[8+3*j]), int(line[9+3*j])])

        if params.single_fixation:
            ### for each image, filter out the gaze point duration <0.1s. Change t (ms) to other numbers to avoid very short fixation points
            gaze_data = filter_gaze(gaze_data, t=350)
            ### for each image, maximal 7 gaze points are allowed. Change N to other numbers for maximal gaze points.
            gaze_data = aggregate_gaze_points(gaze_data, N=7)
        plot_gazemap(params, index2image[image_id], image_id, 1920, 1080, int(line[3]), int(line[4]), int(line[5]), int(line[6]), int(line[1]), int(line[2]), gaze_data)