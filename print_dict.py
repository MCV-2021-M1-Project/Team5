"""
Usage:
  print_dict.py <dictName> 
  print_dict.py -h | --help
Options:
"""
import pickle


if __name__ == "__main__":

    # read args
    dict_name = 'gt_corresps1.pkl'


    with open(dict_name, 'rb') as fd:
        dict = pickle.load(fd)

    print (dict)
    # newDict = []
    # for box in dict:
    #   newDict.append([box])
    # print (newDict)

    # with open('text_boxes_fixed.pkl', 'wb') as fd:
    #     pickle.dump(newDict, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print ('Dictionary contains {} elements'.format(len(dict)))
