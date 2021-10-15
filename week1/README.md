# Team5 
## M1 Content Based Image Retrieval
To learn the basic concepts and techniques to build a simple query by example
retrieval system for finding paintings in a museum image collection.

## Week 1

Image descriptor: image gray-level / color histograms (1D)

### Usage

``
$ task1.py [-p path] [-q query_image] [-f query_image_folder] [-k k_best] [-c color_space] [-plt plot_result]
``

Options:

  -p, --path            relative path to dataset folder
  
  -q, --query_image     relative path to the query image
  
  -f, --query_image_folder  
                        relative path to the folder containing the query images
                        
  -k, --k_best          number of images to retrieve
  
  -c, --color_space     color space to use: "Gray", "RGB", "Lab", "HSV", "YCrCb"
  
  -g, --gt_results      relative path to the query ground truth result
  
  -r,--computed_results 
                        relative path to the computed results (pickle format as specified in the slides)
  
  -v, --validation_metrics
                        set to true to just extract the map@K score
                        
  -m, --mask            set True to remove background from query images

  -plt, --plot_result   set to True to plot results

### Output
After running the script in with the folder argument set, five pkl will be generated
following the format specfied in the slides. Each file correspond to the result for a different similarity/distance method in the color space passed as argument. The name format is:
  - DistanceMetric_ColorSpace.pkl

## Example
Query a single image and display output

``
$ Python3 task1.py -p "./BBDDr" -c "Lab" -q "./qsd1_w1/00000.jpg" -k 5 -c "Lab" -plt True
``

Query images from a folder

``
$ Python3 task1.py -p "./BBDD" -f "./qsd1_w1.jpg" -k 5 -c "Lab"
``

## Testing an especific pickle file
Print the map@k for the passed result file

``
$ Python3 task1.py -v True -g path/to/gt/results.pkl -r path/to/computed/results.pkl -k <number_to_test>
``
