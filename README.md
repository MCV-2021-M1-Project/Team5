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
                        relative path to the computed results
  
  -v, --validation_metrics
                        set to true to extract the metrics
                        
  -m, --mask            set True to remove background

  -plt, --plot_result   set to True to plot results

## Example
Query a single image and display output

``
$ Python3 task1.py -p "./BBDDr" -c "Lab" -q "./qsd1_w1/00000.jpg" -k 5 -c "Lab" -plt True
``

Query images from a folder

``
$ Python3 task1.py -p "./BBDD" -f "./qsd1_w1.jpg" -k 5 -c "Lab"
``
