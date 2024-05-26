import os
import cv2
import sys



def find_fps(filename):

  video = cv2.VideoCapture(filename);

  # Find OpenCV version
  (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

  if int(major_ver)  < 3 :
      fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
      print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  else :
      fps = video.get(cv2.CAP_PROP_FPS)
      print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

  video.release();
  return(fps)

CAP_PROP_POS_MSEC=0 #Current position of the video file in milliseconds.

def frame_iterator(filename, max_num_frames):
  """Uses OpenCV to iterate over all frames of filename at a given frequency.

  Args:
    filename: Path to video file (e.g. mp4)
    every_ms: The duration (in milliseconds) to skip between frames.
    max_num_frames: Maximum number of frames to process, taken from the
      beginning of the video.

  Yields:
    RGB frame with shape (image height, image width, channels)
  """
  fps = find_fps(filename)
  # every_ms = round(1000/fps) #rounding seems to be important. In the first video, without rounding I get 356, but with rounding I get the correct number of 450.
  # every_ms = 1000/fps
  # print(every_ms, 1000/fps)

  video_capture = cv2.VideoCapture()
  if not video_capture.open(filename):
    print >> sys.stderr, 'Error: Cannot open video file ' + filename
    return
  last_ts = -99999  # The timestamp of last retrieved frame.
  num_retrieved = 0

  while num_retrieved < max_num_frames:
    # Skip frames
    # while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
      # if not video_capture.read()[0]:
        # return

    last_ts = video_capture.get(CAP_PROP_POS_MSEC)
    has_frames, frame = video_capture.read()
    if not has_frames:
      break
    yield frame
    num_retrieved += 1


# just a workaround on the shutil.copytree to copy all files from a folder to another: https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth#12514470
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

if __name__ == '__main__':

  # I want to extract the frames from the originally downloaded videos and put them in my directory so that they will not be backed up. I will also copy the annotations to make it easier to use later.
  original_directory           = "/mnt/data_8T/datasets/DHF1k/video/"
  annotation_directory_preffix = "/mnt/data_8T/datasets/DHF1k/annotation/"
  annotation_directory_suffix  = "/maps" #there is also a fixation folder, the name of the folder between the suffix and the preffix has one extra zero compared to the video file name
  video_files      = os.listdir(original_directory)
  annotation_files = os.listdir(annotation_directory_preffix)
  extracted_frames_directory   = "/mnt/data_8T/datasets/DHF1k_extracted/frames"
  copied_annotations_directory = "/mnt/data_8T/datasets/DHF1k_extracted/maps"
  if not os.path.exists(extracted_frames_directory):
    os.makedirs(extracted_frames_directory)

  if not os.path.exists(copied_annotations_directory):
    os.makedirs(copied_annotations_directory)

  check = 0
  for video_file in video_files:
    # The video is named something like "001.AVI", i want the number to match it to the corresponding annotation which will be 0001
    number_of_file = int(video_file.split(".")[0])
    # My working directory path. Each video will have a folder of its own.
    path_to_extracted_frames = os.path.join(extracted_frames_directory, str(number_of_file))
    if not os.path.exists(path_to_extracted_frames):
      os.mkdir(path_to_extracted_frames)

    if number_of_file < 701: #300 Videos have no annotations and are held out by the authors
      for annotation_file in annotation_files:
        # Match the annotation to the video
        if number_of_file == int(annotation_file):
          matched_annotation = annotation_file
          break

      path_to_copied_annotations = os.path.join(copied_annotations_directory, str(number_of_file))
      if not os.path.exists(path_to_copied_annotations):
        os.mkdir(path_to_copied_annotations)

      # Copy the annotations
      import shutil
      source_directory = annotation_directory_preffix + matched_annotation +  annotation_directory_suffix #os path join didnt work for some reason here
      copytree(source_directory, path_to_copied_annotations)

    count = 0
    for frame in frame_iterator(os.path.join(original_directory, video_file), max_num_frames=10000 ):
        count+=1
        #print("extracted {} frame with shape {}".format(count,frame.shape))
        path_to_new_frame = os.path.join(path_to_extracted_frames, str(count)+".png")
        cv2.imwrite(path_to_new_frame, frame)

        """
        import matplotlib.pyplot as plt
        plt.imshow(frame)
        plt.show()
        """
        #It works!

    if number_of_file < 701:
      print("Asserting annotations and frames length is equal for {}...".format(number_of_file))
      # assert(len(os.listdir(path_to_extracted_frames)) == len(os.listdir(path_to_copied_annotations)))
      print(len(os.listdir(path_to_extracted_frames)) , len(os.listdir(path_to_copied_annotations)))
    print("{} done!".format(number_of_file))

    #Checked at 001 and 278, the number of annotations and files match.

