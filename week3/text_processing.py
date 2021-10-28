import glob
import string


def getImagesGtText(path):
    filenames = [img for img in glob.glob(path + "/*"+ ".txt")]
    filenames.sort()
    ddbb_texts = {}
    print(filenames)
    for ind, filename in enumerate(filenames):
        file1 = open(filename, 'r')
        Lines = file1.readlines()
        ddbb_texts[filename.replace('.txt', '.jpg')] = []
        for i, line in enumerate(Lines):
            ddbb_texts[filename.replace('.txt', '.jpg')].append(readTextFromFile(line))

    return ddbb_texts


def readTextFromFile(text):
    "".join(filter(lambda char: char in string.printable, text))

    painter_name = text.split(",", 1)[0]

    if painter_name.count("'") < 2:
        painter_name = (painter_name.split("\""))[1].split("\"")[0]
        ''.join(painter_name.split())
    else:
        painter_name = (painter_name.split("'"))[1].split("'")[0]
        ''.join(painter_name.split())
    print("Ground Truth Text: " + painter_name)

    painting_name = text.split(",", 1)[1]
    if painting_name.count("'") < 2:
        painting_name = (painting_name.split("\""))[1].split("\"")[0]
        ''.join(painting_name.split())
    else:
        painting_name = (painting_name.split("'"))[1].split("'")[0]
        ''.join(painting_name.split())
    print("Ground Truth Text1: " + painting_name)

    return painter_name, painting_name
