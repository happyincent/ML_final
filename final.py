from predict import *
from account import *

import seleniumrequests
from bs4 import BeautifulSoup
from StringIO import StringIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import sys
from keras.models import load_model

SHOW_IMG = False
KEEP_TEST = False
MODEL_FILENAME = 'model_cnn.h5'
INI_FILENAME = 'account.ini'
DRIVER_PATH = './chromedriver'

LOGIN_URL = 'http://selcrs.nsysu.edu.tw/'
CHECK_URL = 'http://selcrs.nsysu.edu.tw/menu4/Studcheck.asp'
WRONG_MSG = "Wrong Validation Code"

def get_vcode(driver):
    vcode_URL = driver.find_element_by_id('imgVC').get_attribute('src')

    vcode_res = driver.request('GET', vcode_URL)
    img = Image.open(StringIO(vcode_res.content))
    
    if SHOW_IMG:
        img.show()
    return img

def convert_vcode(img):
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.ModeFilter)
    img = img.convert('L')

    vcode = []

    for j in range(0, 4):
        quater = img.crop((img.width/4*j, 0,
                          img.width/4*(j+1), img.height))
        quater = np.asarray(quater, dtype="uint8")
        quater.flags.writeable = True

        threshold = np.unique(quater)[0]
        quater[quater > threshold] = 255
        quater[quater == threshold] = 0

        vcode.append(quater)

    vcode = np.asarray(vcode)
    return vcode

def login(driver, account, vcode_str):
    driver.find_element_by_name('stuid').send_keys(account['user_account'])
    driver.find_element_by_name('SPassword').send_keys(account['user_password'])
    driver.find_element_by_name('ValidCode').send_keys(vcode_str)
    driver.find_element_by_name('B1').click()

    login_res = driver.request('GET', CHECK_URL)
    login_res.encoding = 'big5'

    soup = BeautifulSoup(login_res.text, 'html.parser')
    return not(
        soup.find(string=WRONG_MSG) == WRONG_MSG
    )


def main():
    model_name = MODEL_FILENAME
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    print('Use model: ' + model_name)
    model = load_model(model_name)

    account_info = get_account_from_ini(INI_FILENAME)
    print('ID: "' + account_info['user_account'] + '" start login test ...')

    for i in xrange(sys.maxint):
        driver = seleniumrequests.Chrome(DRIVER_PATH)
        driver.get(LOGIN_URL)

        img = get_vcode(driver)
        vcode = convert_vcode(img)

        print('Start predict (' + str(i) + ')!')
        vcode_str = predict(vcode, model_name, model)
        print('Get vcode: "' + vcode_str + '"')
        if_login = login(driver, account_info, vcode_str)
        print('Result: ' + str(if_login))

        if not if_login and not SHOW_IMG:
            ImageOps.invert(img)\
            .filter(ImageFilter.ModeFilter)\
            .convert('L')\
            .show()

        if not if_login or KEEP_TEST:
            driver.close()
        else:
            break

    raw_input('Click ENTER to close!!')


if __name__ == '__main__':
    main()