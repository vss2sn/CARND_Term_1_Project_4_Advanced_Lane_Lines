import glob
import pickle
import matplotlib as mp
import numpy as np
from moviepy.editor import VideoFileClip as vfc
import cv2

def my_camera_calibration(path, nx = 9, ny = 6):
    cal_images = glob.glob(path + '/calibration*.jpg')
    #print(cal_images)
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    image_points, object_points = [], []
    for img_path in cal_images:
        img = mp.image.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            image_points.append(corners)
            object_points.append(objp)
    #print(object_points)
    #print("")
    #print(image_points)
    calibration_data = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    calculate_perspective_matrices(calibration_data)
    pickle.dump(calibration_data, open(path + '/calibration.p', 'wb'))

def load_my_camera_calibration(path='./camera_cal'):
    return pickle.load(open(path + '/calibration.p', 'rb'))

def correct_distortion(img, mtx, dist=None):
    return cv2.undistort(img, mtx, dist, None, mtx)

def color_threshold(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_mask = np.zeros_like(s_channel)
    s_mask[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_mask

def abs_sobel_threshold(img, sobel_kernel = 3, threshx = (0,255), threshy = (0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    gray = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    gray = np.abs(gray)
    gray = np.uint8(255 * (gray/np.max(gray)))
    mask = np.zeros_like(gray)
    mask[(gray >= threshx[0]) & (gray < threshx[1])] = 1
    mask[(gray >= threshy[0]) & (gray < threshy[1])] = 1
    return mask

def mag_and_dir_threshold(img, sobel_kernel=3, mag_thresh=(0, 255),dir_thresh=(0, np.pi / 2)):#Calculates the magnitude of the sobel derivative and returns threshed values
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0,  ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # The magnitude is the square-root of the squared sum
    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)
    sobel_magnitude = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    sobel_magnitude = np.uint8(255 * (sobel_magnitude / np.max(sobel_magnitude)))
    sobel_direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    # Binarize magnitude
    sobel_mask = np.zeros_like(sobel_magnitude)
    sobel_mask[(sobel_magnitude > mag_thresh[0]) & (sobel_magnitude <= mag_thresh[1]) & (sobel_direction > dir_thresh[0]) & (sobel_direction <= dir_thresh[1])] = 1
    return sobel_mask

def combine_sobel_thresholds(img):  #Combines the sobel derivative, magnitude and direction:
    grad = abs_sobel_threshold(img, sobel_kernel = 13, threshx = (20, 255), threshy = (60, 255))
    mag_binary = mag_and_dir_threshold(img, sobel_kernel = 13, mag_thresh = (30, 100), dir_thresh = (-1.57,1.33))
    color_binary = color_threshold(img, thresh = (100, 255))
    combined = np.zeros_like(mag_binary)
    combined[((grad == 1) & (mag_binary == 1)) | (color_binary == 1)] = 1
    return combined

def calculate_perspective_matrices(calibration_data): #    Performs a perspective transformation on given image
    test_images = glob.glob('./test_images/*.jpg')
    images = []
    for img_path in test_images:
        undistorted = correct_distortion(mp.image.imread(img_path),mtx=calibration_data[1],dist=calibration_data[2])
        images.append(undistorted)
    img=images[0]
    im_shape = (img.shape[1], img.shape[0])
    #if src is None and dst is None:
    src = np.float32([[293, 668], [587, 458], [703, 458], [1028, 668]])
    dst = np.float32([[310, im_shape[1]], [310, 0], [950, 0], [950, im_shape[1]]])
    T = cv2.getPerspectiveTransform(src, dst)
    Tinv = cv2.getPerspectiveTransform(dst, src)
    pickle.dump([T, Tinv], open('./perspective_transform.p', 'wb'))

def load_perspective_matrices():
    T, Tinv = pickle.load(open('./perspective_transform.p', 'rb'))
    return T, Tinv

def transform_to_top_view(img, T):
    im_shape = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, T, im_shape)


def fit_polynomial(img, nwindows=9, plotit=False):
    bw_img = img
    hist = np.sum(bw_img[int(bw_img.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((bw_img, bw_img, bw_img)) * 255
    midpoint = np.int(hist.shape[0] / 2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    window_height = np.int(bw_img.shape[0] / nwindows)
    nonzero = bw_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    l_lane_inds = []
    r_lane_inds = []

    for window in range(nwindows):
        win_y_low = bw_img.shape[0] - (window + 1) * window_height
        win_y_high = bw_img.shape[0] - window * window_height
        win_xl_low = leftx_current - margin
        win_xl_high = leftx_current + margin
        win_xr_low = rightx_current - margin
        win_xr_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xl_low, win_y_low),
                      (win_xl_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xr_low, win_y_low),
                      (win_xr_high, win_y_high), (0, 255, 0), 2)
        good_l_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xl_low) & (nonzerox < win_xl_high)).nonzero()[0]
        good_r_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xr_low) & (nonzerox < win_xr_high)).nonzero()[0]
        l_lane_inds.append(good_l_inds)
        r_lane_inds.append(good_r_inds)
        if len(good_l_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_l_inds]))
        if len(good_r_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_r_inds]))
    l_lane_inds = np.concatenate(l_lane_inds)
    r_lane_inds = np.concatenate(r_lane_inds)
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]
    l_fit = np.polyfit(lefty, leftx, 2)
    r_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, bw_img.shape[
                        0] - 1, bw_img.shape[0])
    l_fitx = l_fit[0] * ploty**2 + l_fit[1] * ploty + l_fit[2]
    r_fitx = r_fit[0] * ploty**2 + r_fit[1] * ploty + r_fit[2]

    out_img[nonzeroy[l_lane_inds], nonzerox[
        l_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[r_lane_inds], nonzerox[
        r_lane_inds]] = [0, 0, 255]

    if plotit:
        plt.imshow(out_img / 255)
        plt.plot(l_fitx, ploty, color='yellow')
        plt.plot(r_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return l_fit, r_fit, (leftx, lefty, rightx, righty, ploty)

def calculate_curvature(img, l_fit, r_fit, fits): #    Calculates curvature for given polynomial fits
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = img.shape[0]
    #print(y_eval)
    #y_eval = 0
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    l_fit_cr = np.polyfit(fits[1] * ym_per_pix, fits[0] * xm_per_pix, 2)
    r_fit_cr = np.polyfit(fits[3] * ym_per_pix, fits[2] * xm_per_pix, 2)
    #print(l_fit_cr)
    #print(r_fit_cr)
    l_curverad = ((1 + (2 * l_fit_cr[0] * y_eval * ym_per_pix + l_fit_cr[1])**2)**1.5) / np.absolute(2 * l_fit_cr[0])
    r_curverad = ((1 + (2 * r_fit_cr[0] * y_eval * ym_per_pix + r_fit_cr[1])**2)**1.5) / np.absolute(2 * r_fit_cr[0])
    #print(y_eval*ym_per_pix)
    x_offset = (( (l_fit_cr[0] + r_fit_cr[0])*(y_eval*ym_per_pix)**2 + (l_fit_cr[1] + r_fit_cr[1])*(y_eval*ym_per_pix) + l_fit_cr[2] + r_fit_cr[2]) / 2) - (img.shape[1]*xm_per_pix/2)
    #print(x_offset)
    #x_offset = ((l_fit_cr[2] + r_fit_cr[2]) / 2) - (img.shape[1]*xm_per_pix/2)
    #print(x_offset)
    #print((l_fit_cr[1] + r_fit_cr[1])*(y_eval*ym_per_pix)/2)
    return l_curverad, r_curverad, x_offset


def warp_perspective_back(warped, undist, l_fit, r_fit, fits, Tinv): # Warps perspective back to original view and draws lane area
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    l_fitx = l_fit[0] * fits[4]**2 + l_fit[1] * fits[4] + l_fit[2]
    r_fitx = r_fit[0] * fits[4]**2 + r_fit[1] * fits[4] + r_fit[2]
    pts_left = np.array([np.transpose(np.vstack([l_fitx, fits[4]]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([r_fitx, fits[4]])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(
        color_warp, Tinv, (undist.shape[1], undist.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def set_globals():
    global T_g, Tinv_g, mtx_g, dist_g, l_fit_g, r_fit_g, curvature_g
    ret, mtx, dist, rvecs, tvecs = load_my_camera_calibration('./camera_cal')
    T, Tinv = load_perspective_matrices()
    l_fit_g, r_fit_g, curvature_g = [], [], []
    T_g, Tinv_g, mtx_g, dist_g = T, Tinv, mtx, dist

def process_video(path, f_out):
    #set_globals()
    output = f_out
    clip1 = vfc(path)
    clip = clip1.fl_image(process_frame)
    clip.write_videofile(output, audio=False)

def smooth_curvature(curvature, n=50):
    curvature_g.append(curvature)
    curvature_np = np.array(curvature_g)
    if len(curvature_g) > n:
        curvature = np.mean(curvature_np[-n:])
    return curvature

def smooth_fits(l_fit, r_fit, n=20):
    l_fit_g.append(l_fit)
    r_fit_g.append(r_fit)
    l_fit_np = np.array(l_fit_g)
    r_fit_np = np.array(r_fit_g)
    if len(l_fit_g) > n:
        l_fit = np.mean(l_fit_np[-n:, :], axis=0)
    if len(r_fit_g) > n:
        r_fit = np.mean(r_fit_np[-n:, :], axis=0)
    return l_fit, r_fit

def process_frame(img):
#    cv2.imwrite( "./output_images/image.jpg", img)
    undistorted = correct_distortion(img, mtx=mtx_g, dist=dist_g)
    binary = combine_sobel_thresholds(undistorted)
    perspective_transformed = transform_to_top_view(binary, T=T_g)
    l_fit, r_fit, fits = fit_polynomial(perspective_transformed, plotit=False, nwindows=15)
    l_fit, r_fit = smooth_fits(l_fit, r_fit)
    l_curvature_in_m,r_curvature_in_m,offset_from_center = calculate_curvature(perspective_transformed, l_fit, r_fit, fits)
    #print(l_curvature_in_m)
    #print(r_curvature_in_m)
    result = warp_perspective_back(perspective_transformed, img, l_fit, r_fit, fits, Tinv=Tinv_g)
    curvature_in_m = (l_curvature_in_m+r_curvature_in_m)/2
    curvature_in_m = smooth_curvature(curvature_in_m)
    curvature_text = 'Curvature : {:.2f}'.format(curvature_in_m)
    offset_text = 'Offset from Center : {:.2f}'.format(offset_from_center)
    cv2.putText(result, curvature_text, (200, 100), 0, 1.2, (255, 255, 0), 2)
    cv2.putText(result, offset_text, (200, 200), 0, 1.2, (255, 255, 0), 2)
#    cv2.imwrite( "./output_images/undistorted.jpg", undistorted)
#    cv2.imwrite( "./output_images/binary.jpg", binary*255)
#    cv2.imwrite( "./output_images/perspective_transformed.jpg", perspective_transformed*255)
#    cv2.imwrite( "./output_images/result.jpg", result)
#    exit()
    return result

# REF: https://github.com/CYHSM/carnd/tree/master/CarND-Advanced-Lane-Lines
def main():
    #my_camera_calibration('./camera_cal/')
    set_globals()
    process_video('./project_video.mp4',f_out='out2.mp4')
    #process_video('./challenge_video.mp4', f_out='challenge_out.mp4')

if __name__ == "__main__":
    main()
