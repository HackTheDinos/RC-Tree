from PIL import Image
import pytesseract
import cv2
import sys


def imgToStr(im_image):

    gray_im = cv2.cvtColor(im_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray',gray_im)
    gray_image = Image.fromarray(gray_im)

    tess_configs = "--user-words ./config/user-words.txt " + \
        "--user-patterns ./config/user-patterns.txt " + \
        "--tessedit_char_whitelist " + \
        "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz_0987654321"

    text = pytesseract.image_to_string(gray_image)  # , config=tess_configs)

    return text


def main(file):
    im = cv2.imread(file)
    # cv2.imwrite('images/phrase_orig.png', im)
    # im_image = Image.fromarray(im)

    # im_image.show()

    print(imgToStr(im))


if __name__ == '__main__':
    file = sys.argv[1]
    main(file)

# file_word = 'images/sm_phrase.png'
# file_word = 'images/rb_phrase.png'
# file_word = 'images/mm_phrase.png'
# file_word = 'images/bj_phrase.png'
# file_word = 'images/gp_phrase.png'
# file_word = 'images/fa_phrase.png'
# file_word = 'images/al_phrase.png'
# file_word = 'images/tm_phrase.png'


